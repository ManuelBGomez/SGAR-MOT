"""Mine the frames where the detector fails, on MOTSynth -- step 1 of building the MIR training set.
"""

# Importing libraries:
import argparse
import os
import torch
import cv2
import json

import numpy as np

from tqdm import tqdm

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data import ValTransform
from yolox.tracker import matching
from yolox.tracker.byte_tracker import STrack
from pycocotools.mask import decode

def main(exp, args):
    out_dir = os.path.join(args.data_path, "annotations", "unmatched_detections_per_id")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # initialize detector:
    model = exp.get_model()
    model.cuda()
    model.eval()

    # Load the state dict:
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if args.ckpt is None:
        ckpt = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt = args.ckpt

    # Load checkpoint after establishing it:
    ckpt = torch.load(ckpt, map_location="cuda:0")
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)

    # Preprocessing transform:
    preproc_img = ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # Load MOTSynth dataset:
    videos = os.listdir(os.path.join(args.data_path, "data"))
    # Remove from videos list those already present on output directory:
    videos = [video for video in videos if video + ".json" not in os.listdir(out_dir)]

    # For each video -- processing frame by frame:
    for video in videos:
        print("Processing video: ", video)
        not_tracked_json = []
        # Get the frames:
        frames = os.listdir(os.path.join(args.data_path, "data", video, "rgb"))
        frames.sort()
        # Annotation file:
        anns_file = os.path.join(args.data_path, "data", video, "gt", "gt.txt")
        anns_mask_file = os.path.join(args.data_path, "data", video, "gt", "gt_mask.txt")
        # Load each type of annotations:
        anns_box = np.loadtxt(anns_file, dtype=np.float32, delimiter=",")
        anns_mask = np.loadtxt(anns_mask_file, dtype=np.string_, delimiter=" ")

        # For each frame:
        for i, frame in enumerate(tqdm(frames)):
            # Frame enumeration will start at 0, we will begin at 1:
            frame_id = i + 1
            # Load image:
            img = cv2.imread(os.path.join(args.data_path, "data", video, "rgb", frame))
            original_image_shape = img.shape[:2]

            # Preprocess image:
            img, _ = preproc_img(img, exp.test_size, exp.test_size)
            img = torch.tensor(img).unsqueeze(0).cuda()

            with torch.no_grad():
                outputs = model(img)
                outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)[0]

                # Get the bounding boxes: 
                if outputs is None or len(outputs) == 0:
                    # Skip frame if there are no dets
                    continue
                outputs = outputs.cpu()
                output_bboxes = outputs[:, 0:4]
                scores = outputs[:, 4] * outputs[:, 5]
                scale = min(
                    exp.test_size[0] / original_image_shape[0],
                    exp.test_size[1] / original_image_shape[1],
                )

                output_bboxes /= scale

            # Take information from the current frame (and except ignore regions -- confidence score 1):
            # Filter using the two conditions
            anns_frame = anns_box[np.where(anns_box[:, 0] == frame_id)]
            anns_frame = anns_frame[anns_frame[:, 6] == 1]
            # Ground-truth bboxes:
            gt_bboxes_frame = anns_frame[:, 2:6]
            anns_mask_frame = anns_mask[np.where(anns_mask[:, 0].astype(int) == frame_id)]

            # Get the bounding boxes with confidence over 
            hc_indices = scores > args.track_threshold
            low_indices = scores > 0.1
            high_indices = scores < args.track_threshold
            lc_indices = np.logical_and(low_indices, high_indices)

            # First association: high confidence detections with ground-truth bboxes
            detections = [STrack(STrack.tlbr_to_tlwh(bbox), score) for bbox, score in zip(output_bboxes[hc_indices], scores[hc_indices])]
            gt_bboxes = [STrack(bbox, 1) for bbox in gt_bboxes_frame] # GT bbox is already in tlwh format

            dists = matching.iou_distance(gt_bboxes, detections)
            matches, u_gt, u_detection = matching.linear_assignment(dists, thresh=args.match_thresh)

            # Second association: low confidence detections with ground-truth bboxes
            detections_second = [STrack(STrack.tlbr_to_tlwh(bbox), score) for bbox, score in zip(output_bboxes[lc_indices], scores[lc_indices])]
            gt_bboxes_second = [gt_bboxes[counter] for counter in u_gt]
            gt_full_second = [anns_frame[counter] for counter in u_gt] # Full annotation info - we'll need it when recovering information of undetected ground-truth bboxes

            dists_second = matching.iou_distance(gt_bboxes_second, detections_second)
            matches_second, u_gt_second, u_detection_second = matching.linear_assignment(dists_second, thresh=args.match_thresh_2nd)

            # We will take now unmatched ground-truth bboxes:
            unmatched_gt = [gt_full_second[counter] for counter in u_gt_second]



            # We will iterate over each unmatched ground_truth, checking visibility:
            for gt in unmatched_gt:
                # We will check firstly if it exists this object in the mask file:
                gt_mask_info = anns_mask_frame[anns_mask_frame[:, 1].astype(int) == gt[1] + 2000]
                if len(gt_mask_info) == 0:
                    continue # Ignore this unmatched gt as there is no mask associated.
                # If we have the mask, we take it with the bounding box and check visibility:
                gt_mask = np.array(decode({"size": [int(gt_mask_info[0, 3]), int(gt_mask_info[0, 4])], 
                                           "counts": gt_mask_info[0, 5].decode("utf-8")}), dtype=bool)
                gt_bbox = gt[2:6]

                # We will check the visibility of the object (area_mask / area_bbox):
                area_mask = np.sum(gt_mask)
                area_bbox = gt_bbox[2] * gt_bbox[3]
                visibility = area_mask / area_bbox
                # If not enough, we ignore this object:
                if visibility < args.visibility_threshold:
                    continue

                # If we reach this point, we will consider this object, in this frame, as tracked:
                # Check if there is the object with this ID in the json:
                object_id = gt[1]
                # not_tracked_json will be an array with format {track_id, list_frames_not_matched}.
                # If the object_id is not in the list, we will add it:
                if len(not_tracked_json) == 0 or object_id not in [elem["object_id"] for elem in not_tracked_json]:
                    not_tracked_json.append({"object_id": int(object_id), "list_frames_not_matched": [int(frame_id)]})
                else:
                    # If the object_id is already in the list, we will append the frame_id:
                    not_tracked_json[[elem["object_id"] for elem in not_tracked_json].index(object_id)]["list_frames_not_matched"].append(frame_id)



        # Save JSON file in output directory:
        with open(os.path.join(out_dir, video + ".json"), "w") as f:
            json.dump(not_tracked_json, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get third round points for MOTSynth dataset")
    parser.add_argument("--exp-file", type=str, default=None, help="Path to the experiment file")
    parser.add_argument("-expn", "--experiment-name", type=str, default="MOTSynth-third", help="Experiment name")
    parser.add_argument("-c", "--ckpt", type=str, default=None, help="Path to the checkpoint detector file")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the MOTSynth dataset")
    parser.add_argument("--track_threshold", type=float, default=0.6, help="Threshold for considering high confidence/low confidence detections")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="Threshold for matching detections with ground-truth bboxes")
    parser.add_argument("--match_thresh_2nd", type=float, default=0.5, help="Threshold for matching low confidence detections with ground-truth bboxes")
    parser.add_argument("--visibility_threshold", type=float, default=0.05, help="Threshold for considering that an object is occluded (not visible enough)")

    args = parser.parse_args()
    exp = get_exp(args.exp_file, None)

    main(exp, args)
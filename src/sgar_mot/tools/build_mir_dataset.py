"""Script that allows to build the MIR training set -- step 2, after `tools/mine_unmatched_detections.py`.

For each frame where the detector failed (mined in step 1), this generates the three
time-aligned masks with SAM 2 (detection frame, previous frame and current frame)
embeds them, and stores one sample per situation.
"""

import os
import json
import pickle
import numpy as np
import cv2
import torch

from torchmetrics import JaccardIndex
from tqdm import tqdm
from pycocotools.mask import decode

from sgar_mot.orp import embeddings as dcu

# We will use SAM 2 image predictor to extract the features from the images and some masks:
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_MODEL_TYPE = 'sam2_hiera_l.yaml'
SAM_CHECKPOINT = 'pretrained/sam2_hiera_large.pt'
# Frames where the detector failed, produced by tools/mine_unmatched_detections.py:
DATA_PATH = "datasets/MOTSynth/annotations/unmatched_detections_per_id"
# Directory holding the MOTSynth videos:
VIDEO_PATH = "datasets/MOTSynth/data"
# Where the per-video sample pickles are written:
SAVE_PATH = "datasets/mir_dataset"
INITIAL_VIDEO = 501
FINAL_VIDEO = 767
NUM_GROUPS_PER_OBJECT = 6
NUM_SAMPLES_PER_FRAME_GROUP = 3

def main():
    # Set seed in numpy:
    np.random.seed(28)

    data_path, video_path, save_path = DATA_PATH, VIDEO_PATH, SAVE_PATH
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Creation of SAM2 predictor:
    sam2_model = build_sam2(SAM_MODEL_TYPE, SAM_CHECKPOINT, device = "cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # We list all files in the data path:
    files = os.listdir(data_path)
    # Get files already present in save_path:
    files_save = os.listdir(save_path)
    # Replace .pkl extension by json to make comparison:
    files_save = [file.replace(".pkl", ".json") for file in files_save]
    # Remove from files list those already saved in save_path and therefore, we won't process them again:
    files = sorted(list(set(files) - set(files_save)))
    # First, before processing, remove those outside initial and final video range:
    files = [file for file in files if int(file.split(".")[0]) >= INITIAL_VIDEO and int(file.split(".")[0]) <= FINAL_VIDEO]
    
    # Take count of the total number of correct and incorrect matches:
    total_correct = 0
    total_incorrect = 0

    # We iterate over all files:
    for file in files:
        print(f"Processing file {file}")
        objects_info = []

        # We load json data:
        with open(os.path.join(data_path, file), "r") as f:
            data = json.load(f)
            # Get the video number from the file name -- removing the extension:
            video_number = file.split(".")[0]

            # Load annotation files:
            anns_box_path = os.path.join(video_path, video_number, "gt", "gt.txt")
            anns_mask_path = os.path.join(video_path, video_number, "gt", "gt_mask.txt")
            anns_box = np.loadtxt(anns_box_path, dtype=np.float32, delimiter=",")
            anns_mask = np.loadtxt(anns_mask_path, dtype=np.string_, delimiter=" ")

            # Iterate over matches:
            for difficult_matching in tqdm(data):
                # This is the frame in which the object is lost:
                frame_ids = difficult_matching["list_frames_not_matched"]
                tr_id = difficult_matching["object_id"]

                # Split tr_ids:
                frame_ids = divide_frame_array(frame_ids)

                # Select a random group:
                frame_id_group_idxs = np.random.choice(len(frame_ids), NUM_GROUPS_PER_OBJECT if NUM_GROUPS_PER_OBJECT <= len(frame_ids) else len(frame_ids), replace=False)
                for frame_id_group_idx in frame_id_group_idxs:
                    frame_id_group = frame_ids[frame_id_group_idx]

                    # For each group, we will generate one example:
                    # for frame_id_group in frame_ids:

                    # Choose a random frame from the frame group:
                    frame_id_idxs = np.random.choice(len(frame_id_group), NUM_SAMPLES_PER_FRAME_GROUP if NUM_SAMPLES_PER_FRAME_GROUP <= len(frame_id_group) else len(frame_id_group), replace=False)
                    for frame_id_idx in frame_id_idxs:
                        frame_id = frame_id_group[frame_id_idx]
                        # Get the det id -- (ORIGINALLY: first element of the array minus one --> "Last detection available"; NOW: previous frame to the last detection available).
                        det_id = frame_id_group[0] - 2
                        # Select with a random number the frame to do idsw:
                        frame_to_idsw = np.random.choice([0, 1, 2], 1)[0]
                        # Select with a random number the frame to displace:
                        frame_to_displ = np.random.choice([0, 1, 2], 1)[0]
                        # print("Frame to IDSW:", frame_to_idsw, ". Frame to Displ:", frame_to_displ)
                        # Det id and prev_frame_id may coincide, that shouldn't be a problem.
                        first_sample, second_sample, third_sample = create_sample_set(anns_box, anns_mask, video_path, video_number, predictor, tr_id, frame_id, det_id,
                                                                                      idsw_det_frame = frame_to_idsw == 0, idsw_tm1_frame = frame_to_idsw == 1, idsw_t_frame = frame_to_idsw == 2,
                                                                                      displ_det_frame = frame_to_displ == 0, displ_tm1_frame = frame_to_displ == 1, displ_t_frame = frame_to_displ == 2)
                        if first_sample is not None and second_sample is not None and third_sample is not None:
                            total_correct += 3 # We extract three samples per one correct match.
                            objects_info.append(first_sample)
                            objects_info.append(second_sample)
                            objects_info.append(third_sample)
                        else:
                            total_incorrect += 1
            
            # Save objects_info as a pickle file in the output destination:
            with open(os.path.join(save_path, f"{video_number}.pkl"), "wb") as f:
                pickle.dump(objects_info, f)
            # break # Debugging

    print(f"Total correct: {total_correct}")
    print(f"Total incorrect: {total_incorrect}")
    print(f"Total difficult matches: {total_correct + total_incorrect}")

def create_sample_set(anns_box, anns_mask, video_path, video_number, predictor, tr_id, frame_id, det_id,
                      idsw_det_frame = False, idsw_tm1_frame = False, idsw_t_frame = False,
                      displ_det_frame = False, displ_tm1_frame = False, displ_t_frame = False):
    """Function that creates a positive and a negative example for a given track id and frame id.

    Parameters
    ----------
        - anns_box: np.array
            Annotations of the bounding boxes.
        - anns_mask: np.array
            Annotations of the masks.
        - video_path: str
            Path to the video.
        - video_number: str
            Number of the video.
        - predictor: SAM2ImagePredictor
            SAM2 predictor.
        - tr_id: int
            Track id.
        - frame_id: int
            Frame id.
        - det_id: int
            Detection id.
        - idsw_det_frame: bool
            Boolean that indicates if we want to apply IDSW to the detection frame.
        - idsw_tm1_frame: bool
            Boolean that indicates if we want to apply IDSW to the previous frame.
        - idsw_t_frame: bool
            Boolean that indicates if we want to apply IDSW to the current frame.
        - displ_det_frame: bool
            Boolean that indicates if we want to displace the bounding box in the detection frame.
        - displ_tm1_frame: bool
            Boolean that indicates if we want to displace the bounding box in the previous frame.
        - displ_t_frame: bool
            Boolean that indicates if we want to displace the bounding box in the current frame.
    Returns
    -------
        - positive_sample: dict
            Positive example.
        - negative_sample: dict
            Negative example.
    """

    # ---- ANNOTATIONS EXTRACTION ----
    # Get the object information (in current frame):
    curr_frame_anns_box = anns_box[np.where(anns_box[:, 0].astype(int) == frame_id)]
    curr_frame_anns_mask = anns_mask[np.where(anns_mask[:, 0].astype(int) == frame_id)]
    # Get now the information of that track id:
    track_anns_box_curr = curr_frame_anns_box[np.where(curr_frame_anns_box[:, 1].astype(int) == tr_id)]
    # Remember that mask annotations track id is like the bbox annotations track id but adding 2000:
    track_anns_mask_curr = curr_frame_anns_mask[np.where(curr_frame_anns_mask[:, 1].astype(int) == tr_id + 2000)]
    if len(track_anns_mask_curr) == 0:
        # print("Not in current frame. Skipping")
        return None, None, None

    # Get the object information (in previous frame):
    prev_frame_anns_box = anns_box[np.where(anns_box[:, 0].astype(int) == frame_id - 1)]
    prev_frame_anns_mask = anns_mask[np.where(anns_mask[:, 0].astype(int) == frame_id - 1)]
    # Get now the information of that track id:
    track_anns_box_prev = prev_frame_anns_box[np.where(prev_frame_anns_box[:, 1].astype(int) == tr_id)]
    # Remember that mask annotations track id is like the bbox annotations track id but adding 2000:
    track_anns_mask_prev = prev_frame_anns_mask[np.where(prev_frame_anns_mask[:, 1].astype(int) == tr_id + 2000)]
    if len(track_anns_mask_prev) == 0 or len(track_anns_box_prev) == 0:
        # print("Not in previous frame. Skipping")
        return None, None, None

    # Get the object information (in det frame):
    det_frame_anns_box = anns_box[np.where(anns_box[:, 0].astype(int) == det_id)]
    det_frame_anns_mask = anns_mask[np.where(anns_mask[:, 0].astype(int) == det_id)]
    # Get now the information of that track id:
    track_anns_box_det = det_frame_anns_box[np.where(det_frame_anns_box[:, 1].astype(int) == tr_id)]
    # Remember that mask annotations track id is like the bbox annotations track id but adding 2000:
    track_anns_mask_det = det_frame_anns_mask[np.where(det_frame_anns_mask[:, 1].astype(int) == tr_id + 2000)]
    if len(track_anns_mask_det) == 0 or len(track_anns_box_det) == 0:
        # print("Not in detection frame. Skipping")
        return None, None, None

    # For each case, we will get a positive and a negative example:
    # ---- PREVIOUS FRAME (frame_id - 1) ----
    # Get the embedding of the previous frame bbox (we may not use it in the beginning):
    img_path = os.path.join(video_path, video_number, "rgb", f"{frame_id - 1:04d}.jpg")
    img = cv2.imread(img_path)
    # Set image:
    predictor.set_image(img)
    # Get detection bbox:
    bbox_prev = track_anns_box_prev[0][2:6].astype(int)
    bbox_prev[2] = bbox_prev[2] + bbox_prev[0]
    bbox_prev[3] = bbox_prev[3] + bbox_prev[1]
    # Apply slight displacement:
    bbox_prev_adj = displace_bbox(bbox_prev, 1, 5)
    # If the IoU falls below 0.7, use groundtruth directly:
    bbox_prev = bbox_prev if calculate_iou(bbox_prev, bbox_prev_adj) < 0.7 else bbox_prev_adj
    # print("Good" if calculate_iou(bbox_prev, bbox_prev_adj) > 0.7 else "Bad")

    # Crop to visible area:
    bbox_prev[0] = max(0, bbox_prev[0])
    bbox_prev[1] = max(0, bbox_prev[1])
    # As we already have converted it to tlbr format, we will only do the necessary crop, not the format conversion:
    bbox_prev[2] = min(img.shape[1], bbox_prev[2])
    bbox_prev[3] = min(img.shape[0], bbox_prev[3])

    # Generate SAM mask:
    mask_prev, _, _ = predictor.predict(box = bbox_prev, multimask_output=False)
    mask_prev = np.array(mask_prev[0])
    # Call extraction embeddings function:
    gt_mask_prev = np.array(decode({"size": [int(track_anns_mask_prev[0, 3]), int(track_anns_mask_prev[0, 4])], "counts": track_anns_mask_prev[0, 5].decode("utf-8")}), dtype=bool)
    mask_prev_embs, _ = dcu.extract_embeddings(mask_prev, predictor, img)

    # Get the mask from the bounding box:
    mask_bbox_prev = dcu.bbox_as_mask(bbox_prev, mask_prev.shape)
    # Get the embeddings:
    mask_bbox_prev_embs, _ = dcu.extract_embeddings(mask_bbox_prev, predictor, img)

    if idsw_tm1_frame:
        # Generate identity switch mask and detection (we will use aux function):
        bbox_prev_idsw, mask_prev_idsw_embs, mask_prev_idsw = generate_idsw_sample(prev_frame_anns_box, track_anns_box_prev[0][2:6].astype(int), tr_id, predictor, img)
        if bbox_prev_idsw is None:
            return None, None, None
    if displ_tm1_frame:
        # Generate displacement mask and detection (we will use aux function):
        bbox_prev_displ, mask_prev_displ_embs, mask_prev_displ = generate_displ_sample(track_anns_box_prev[0][2:6].astype(int), predictor, img)

    # ---- DETECTION FRAME (det_id) ----
    # Only if previous frame is not the same as detection frame:
    if frame_id - 1 != det_id:
        # Set new image:
        img_path = os.path.join(video_path, video_number, "rgb", f"{det_id:04d}.jpg")
        img = cv2.imread(img_path)
        # Set image:
        predictor.set_image(img)

    # Get detection bbox:
    bbox_det = track_anns_box_det[0][2:6].astype(int)
    bbox_det[2] = bbox_det[2] + bbox_det[0]
    bbox_det[3] = bbox_det[3] + bbox_det[1]
    # Apply slight displacement:
    bbox_det_adj = displace_bbox(bbox_det, 1, 5)
    # If the IoU falls below 0.7, use groundtruth directly:
    bbox_det = bbox_det if calculate_iou(bbox_det, bbox_det_adj) < 0.7 else bbox_det_adj

    # Crop to visible area:
    bbox_det[0] = max(0, bbox_det[0])
    bbox_det[1] = max(0, bbox_det[1])
    # As we already have converted it to tlbr format, we will only do the necessary crop, not the format conversion:
    bbox_det[2] = min(img.shape[1], bbox_det[2])
    bbox_det[3] = min(img.shape[0], bbox_det[3])

    # Generate SAM mask:
    mask_det, _, _ = predictor.predict(box = bbox_det, multimask_output=False)
    mask_det = np.array(mask_det[0])

    # Call extraction embeddings function:
    gt_mask_det = np.array(decode({"size": [int(track_anns_mask_det[0, 3]), int(track_anns_mask_det[0, 4])], "counts": track_anns_mask_det[0, 5].decode("utf-8")}), dtype=bool)
    mask_det_embs, _ = dcu.extract_embeddings(mask_det, predictor, img)

    if idsw_det_frame:
        # Generate identity switch mask and detection (we will use aux function):
        bbox_det_idsw, mask_det_idsw_embs, mask_det_idsw = generate_idsw_sample(det_frame_anns_box, track_anns_box_det[0][2:6].astype(int), tr_id, predictor, img)
        if bbox_det_idsw is None:
            return None, None, None
    if displ_det_frame:
        # Generate displacement mask and detection (we will use aux function):
        bbox_det_displ, mask_det_displ_embs, mask_det_displ = generate_displ_sample(track_anns_box_det[0][2:6].astype(int), predictor, img)

    # ---- CURRENT FRAME (frame_id) ----
    # First --- we get the mask with SAM 2:
    # Load image from current frame:
    img_path = os.path.join(video_path, video_number, "rgb", f"{frame_id:04d}.jpg")
    img = cv2.imread(img_path)
    # Set image:
    predictor.set_image(img)
    # Get curr_frame bbox from annotations:
    curr_bbox = track_anns_box_curr[0][2:6].astype(int)
    curr_bbox[2] = curr_bbox[2] + curr_bbox[0]
    curr_bbox[3] = curr_bbox[3] + curr_bbox[1]
    # Apply slight displacement:
    curr_bbox_adj = displace_bbox(curr_bbox, 1, 5)
    # If the IoU falls below 0.7, use groundtruth directly:
    curr_bbox = curr_bbox if calculate_iou(curr_bbox, curr_bbox_adj) < 0.7 else curr_bbox_adj

    # Crop to visible area:
    curr_bbox[0] = max(0, curr_bbox[0])
    curr_bbox[1] = max(0, curr_bbox[1])
    # As we already have converted it to tlbr format, we will only do the necessary crop, not the format conversion:
    curr_bbox[2] = min(img.shape[1], curr_bbox[2])
    curr_bbox[3] = min(img.shape[0], curr_bbox[3])

    # Get the mask:
    curr_mask, _, _ = predictor.predict(box = curr_bbox, multimask_output=False)
    # Remove first dimension:
    curr_mask = np.array(curr_mask[0])
    # Get ground-truth mask: 
    gt_curr_mask = np.array(decode({"size": [int(track_anns_mask_curr[0, 3]), int(track_anns_mask_curr[0, 4])], "counts": track_anns_mask_curr[0, 5].decode("utf-8")}), dtype=bool)
    # Get embeddings:
    mask_curr_embs, _ = dcu.extract_embeddings(curr_mask, predictor, img)

    if idsw_t_frame:
        # Generate identity switch mask and detection (we will use aux function):
        bbox_curr_idsw, mask_curr_idsw_embs, mask_curr_idsw = generate_idsw_sample(curr_frame_anns_box, track_anns_box_curr[0][2:6].astype(int), tr_id, predictor, img)
        if bbox_curr_idsw is None:
            return None, None, None
    if displ_t_frame:
        # Generate displacement mask and detection (we will use aux function):
        bbox_curr_displ, mask_curr_displ_embs, mask_curr_displ = generate_displ_sample(track_anns_box_curr[0][2:6].astype(int), predictor, img)

    # If groundtruth mask area relation with bbox area is lower than 0.05, we will skip this example:
    bbox_area = (curr_bbox[2] - curr_bbox[0]) * (curr_bbox[3] - curr_bbox[1])
    mask_area = np.sum(gt_curr_mask)
    if mask_area / bbox_area < 0.05:
        # We want to focus on matches with certain visibility, so we will skip this example:
        # print("Too small mask area. Skipping")
        return None, None, None
    
    # Sample decision --> We know that bounding boxes are with IoU > 0.7 ---> No problem.
    # If ANY of the masks has an IoU lower than 0.5 --> Negative example.
    # If ALL of the masks have an IoU greater than 0.5 --> Positive example.
    # Also, we will generate two more negatives: one with displacement and one with identity switch ---> No need to check IoU.
    jac = JaccardIndex("binary")
    iou_det_frame = jac(torch.tensor(mask_det), torch.tensor(gt_mask_det)).item()
    iou_tm1_frame = jac(torch.tensor(mask_prev), torch.tensor(gt_mask_prev)).item()
    iou_t_frame = jac(torch.tensor(curr_mask), torch.tensor(gt_curr_mask)).item()
    output = 1 if iou_det_frame > 0.5 and iou_tm1_frame > 0.5 and iou_t_frame > 0.5 else 0

    first_sample = {
        "frame_idx": frame_id,
        "det_frame_idx": det_id,
        "object_id": tr_id,
        "prev_frame_bbox": np.array(bbox_prev).astype(np.float16),
        "prev_frame_bbox_emb": mask_bbox_prev_embs.astype(np.float16),
        "prev_frame_mask_emb": mask_prev_embs.astype(np.float16),
        "det_frame_bbox": np.array(bbox_det).astype(np.float16),
        "det_frame_mask_emb": mask_det_embs.astype(np.float16),
        "curr_frame_bbox": np.array(curr_bbox).astype(np.float16),
        "curr_frame_mask_emb": mask_curr_embs.astype(np.float16),
        "output": np.array([output]).astype(np.float16),
        "type": 0
    }

    # Second sample with identity switch on any of the objects:
    second_sample = {
        "frame_idx": frame_id,
        "det_frame_idx": det_id,
        "object_id": tr_id,
        "prev_frame_bbox": np.array(bbox_prev_idsw).astype(np.float16) if idsw_tm1_frame else np.array(bbox_prev).astype(np.float16),
        "prev_frame_bbox_emb": mask_bbox_prev_embs.astype(np.float16),
        "prev_frame_mask_emb": mask_prev_idsw_embs.astype(np.float16) if idsw_tm1_frame else mask_prev_embs.astype(np.float16),
        "det_frame_bbox": np.array(bbox_det_idsw).astype(np.float16) if idsw_det_frame else np.array(bbox_det).astype(np.float16),
        "det_frame_mask_emb": mask_det_idsw_embs.astype(np.float16) if idsw_det_frame else mask_det_embs.astype(np.float16),
        "curr_frame_bbox": np.array(bbox_curr_idsw).astype(np.float16) if idsw_t_frame else np.array(curr_bbox).astype(np.float16),
        "curr_frame_mask_emb": mask_curr_idsw_embs.astype(np.float16) if idsw_t_frame else mask_curr_embs.astype(np.float16),
        "output": np.array([0]).astype(np.float16),
        "type": 1
    }

    # Third sample with displacement on any of the objects:
    third_sample = {
        "frame_idx": frame_id,
        "det_frame_idx": det_id,
        "object_id": tr_id,
        "prev_frame_bbox": np.array(bbox_prev_displ).astype(np.float16) if displ_tm1_frame else np.array(bbox_prev).astype(np.float16),
        "prev_frame_bbox_emb": mask_bbox_prev_embs.astype(np.float16),
        "prev_frame_mask_emb": mask_prev_displ_embs.astype(np.float16) if displ_tm1_frame else mask_prev_embs.astype(np.float16),
        "det_frame_bbox": np.array(bbox_det_displ).astype(np.float16) if displ_det_frame else np.array(bbox_det).astype(np.float16),
        "det_frame_mask_emb": mask_det_displ_embs.astype(np.float16) if displ_det_frame else mask_det_embs.astype(np.float16),
        "curr_frame_bbox": np.array(bbox_curr_displ).astype(np.float16) if displ_t_frame else np.array(curr_bbox).astype(np.float16),
        "curr_frame_mask_emb": mask_curr_displ_embs.astype(np.float16) if displ_t_frame else mask_curr_embs.astype(np.float16),
        "output": np.array([0]).astype(np.float16),
        "type": 2
    }

        
    # Return samples:
    return first_sample, second_sample, third_sample

def divide_frame_array(array):
    """Function that divides an array of frame indexes into fragments of consecutive indexes.
    
    Parameters
    ----------
        - array: list
            List of frame indexes.
    Returns
    -------
        - fragmented_array: list
            List of lists with the fragmented frame indexes.
    """
    if len(array) == 0:
        return []
    
    array = np.array(array)
    # Get consecutive differences between elements:
    diffs = np.diff(array)
    # Find where the differences are greater than 1:
    splits = np.where(diffs != 1)[0] + 1
    # Divide the array -- get the fragments:
    fragmented_array = np.split(array, splits)
    
    # Convertimos a listas (si es necesario)
    return [fr.tolist() for fr in fragmented_array]
    
def displace_bbox(bbox, min_displacement, max_displacement):
    """Function that displaces a bounding box randomly a percentage of its width and height.
    
    Parameters
    ----------
        - bbox: np.array
            Bounding box.
        - max_displacement: float
            Maximum displacement percentage.
            
    Returns
    -------
        - new_bbox: np.array
            New bounding box.
    """
    # Get the width and height:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    # Get the displacement:
    min_displacement_x = int(width * min_displacement / 100)
    min_displacement_y = int(height * min_displacement / 100)
    max_displacement_x = int(width * max_displacement / 100)
    max_displacement_y = int(height * max_displacement / 100)
    # Increase max displacement to avoid not moving the bbox:
    if max_displacement_x - min_displacement_x < 1:
        max_displacement_x += 1 
    if max_displacement_y - min_displacement_y < 1:
        max_displacement_y += 1
    # Different displacement for each corner --> In that way we also modify the size:
    dx_min = np.random.randint(min_displacement_x, max_displacement_x)
    dy_min = np.random.randint(min_displacement_y, max_displacement_y)
    # Randomly apply sum or subtraction:
    dx_min = dx_min if np.random.choice([True, False], 1)[0] else -dx_min
    dy_min = dy_min if np.random.choice([True, False], 1)[0] else -dy_min
    # dx_max = np.random.randint(-displacement_x, displacement_x)
    # dy_max = np.random.randint(-displacement_y, displacement_y)
    # Displace the bounding box:
    new_bbox = bbox.copy()
    new_bbox[0] += dx_min
    new_bbox[1] += dy_min
    new_bbox[2] += dx_min
    new_bbox[3] += dy_min
    # Return the new bounding box:
    return new_bbox

def calculate_iou(bbox1, bbox2):
    """Calculate iou between two bounding boxes (tlbr format).
    
    Parameters
    ----------
        - bbox1: np.array
            First bounding box.
        - bbox2: np.array
            Second bounding box.
    Returns
    -------
        - iou: float
            IoU value.
    """
    # Get the intersection:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    # Calculate the area of the intersection:
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # Calculate the area of the union:
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    # Calculate the IoU:
    iou = intersection / union
    # Return the IoU value:
    return iou

def generate_idsw_sample(frame_bbox_anns, original_bbox, track_id, predictor, frame):
    """Function that generates an identity switch sample.
    
    Parameters
    ----------
        - frame_bbox_anns: np.array
            Bounding box annotations of the frame.
        - track_id: int
            Track id.
        - predictor: SAM2ImagePredictor
            SAM2 predictor.

    Returns
    -------
        - bounding box of the identity switched sample
        - mask embedding of the identity switched sample -- we do not need any comparison with ground
        truth as it will be a negative sample in any case.
    """
    # Search for the closest boudning box to the original one:
    closest_bbox = None
    min_distance = float("inf")
    if len(frame_bbox_anns) == 1:
        return None, None, None
    for bbox_ann in frame_bbox_anns:
        # Skip the annotation from the same track id:
        if bbox_ann[1] == track_id:
            continue
        # Get the bbox:
        bbox = bbox_ann[2:6].astype(int)
        # Calculate the distance:
        distance = np.linalg.norm(original_bbox - bbox)
        if distance < min_distance:
            min_distance = distance
            closest_bbox = bbox
    # Put bbox in tlbr format:
    closest_bbox[2] = closest_bbox[2] + closest_bbox[0]
    closest_bbox[3] = closest_bbox[3] + closest_bbox[1]
    # Slighlty displace the bounding box to avoid exact ground-truth case:
    closest_bbox_displ = displace_bbox(closest_bbox, 1, 5)
    # If the IoU falls below 0.7, use groundtruth directly:
    closest_bbox = closest_bbox if calculate_iou(closest_bbox_displ, closest_bbox) < 0.7 else closest_bbox_displ
    # Crop bbox to visible area:
    closest_bbox[0] = max(0, closest_bbox[0])
    closest_bbox[1] = max(0, closest_bbox[1])
    closest_bbox[2] = min(frame.shape[1], closest_bbox[2])
    closest_bbox[3] = min(frame.shape[0], closest_bbox[3])
    # Predict a mask with that displaced bounding box:
    mask_frame, _, _ = predictor.predict(box = closest_bbox, multimask_output=False)
    mask_frame = np.array(mask_frame[0])
    # Get the embeddings:
    mask_frame_embs, _ = dcu.extract_embeddings(mask_frame, predictor, frame)
    # Return the bounding box and the embeddings:
    return closest_bbox, mask_frame_embs, mask_frame

def generate_displ_sample(original_bbox, predictor, frame):
    """Function that generates an identity switch sample.
    
    Parameters
    ----------
        - frame_bbox_anns: np.array
            Bounding box annotations of the frame.
        - track_id: int
            Track id.
        - predictor: SAM2ImagePredictor
            SAM2 predictor.

    Returns
    -------
        - bounding box of the identity switched sample
        - mask embedding of the identity switched sample -- we do not need any comparison with ground
        truth as it will be a negative sample in any case.
    """
    # Put bbox in tlbr format:
    original_bbox[2] = original_bbox[2] + original_bbox[0]
    original_bbox[3] = original_bbox[3] + original_bbox[1]
    # Search for a displaced bounding box --> Apply a severe displacement
    displaced_bbox = displace_bbox(original_bbox, 15, 25)
    # print("IoU original/displaced bbox:", calculate_iou(original_bbox, displaced_bbox))
    # Crop bbox to visible area:
    displaced_bbox[0] = max(0, displaced_bbox[0])
    displaced_bbox[1] = max(0, displaced_bbox[1])
    displaced_bbox[2] = min(frame.shape[1], displaced_bbox[2])
    displaced_bbox[3] = min(frame.shape[0], displaced_bbox[3])
    # Predict a mask with that displaced bounding box:
    mask_frame, _, _ = predictor.predict(box = displaced_bbox, multimask_output=False)
    mask_frame = np.array(mask_frame[0])
    # Get the embeddings:
    mask_frame_embs, _ = dcu.extract_embeddings(mask_frame, predictor, frame)
    # Return the bounding box and the embeddings:
    return displaced_bbox, mask_frame_embs, mask_frame

if __name__ == "__main__":
    main()
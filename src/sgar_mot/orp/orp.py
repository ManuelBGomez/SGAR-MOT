"""The main class representing the ORP module.
"""

import gc
import os
import time

import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from yolox.tracker import matching

from sgar_mot.mir import MIR
from sgar_mot.orp.embeddings import extract_embeddings
from sgar_mot.utils.config import load_args_from_config

# Minimum overlap between an unmatched track and a matched one for the latter to
# be used as context, and for the former to be considered occluded at all.
MIN_OVERLAP = 0.1


class ORP(object):
    def __init__(self, sam_checkpoint, sam_config, mir_config, data_path, video,
                 val_half=False, offload_cpu=False, gamma=0.5):
        """Initialization method.

        Parameters
        ----------
            - sam_checkpoint: path to the SAM 2 weights.
            - sam_config: SAM 2 model configuration file (e.g. sam2_hiera_l.yaml).
            - mir_config: path to the MIR configuration file, which also holds the
              path to the MIR weights.
            - data_path: path to the split of the dataset being tracked.
            - video: name of the sequence being tracked.
            - val_half: whether only the second half of the sequence is evaluated.
            - offload_cpu: whether to offload the video frames to CPU memory.
            - gamma: probability above which MIR recovers a track.
        """
        self.sam_checkpoint = sam_checkpoint
        self.sam_config = sam_config
        self.video = video
        self.data_path = data_path
        self.offload_cpu = offload_cpu
        self.gamma = gamma

        # Video segmentation model -- frozen:
        self.sam2_predictor = build_sam2_video_predictor(self.sam_config, self.sam_checkpoint, device="cuda")
        self.sam2_img_predictor = SAM2ImagePredictor(build_sam2(self.sam_config, self.sam_checkpoint, device="cuda"))

        # Mask Instance Resolver:
        mir_args = load_args_from_config(mir_config).transformer
        self.mir = MIR(mir_args)
        if mir_args.model != "":
            self.mir.load_state_dict(torch.load(mir_args.model))
            print("Loaded MIR weights from:", mir_args.model)
            print("MIR parameters: {:,} ({:.2f} M)".format(self.mir.num_params, self.mir.num_params / 1e6))
        self.mir.eval()

        if video is not None:
            self.video_folder = "rgb" if "MOTSynth" in data_path else "img1"
            self.initial_frame = len(os.listdir(os.path.join(data_path, video, self.video_folder))) // 2 + 1 if val_half else 0
            self.sam2_state = self.sam2_predictor.init_state(os.path.join(data_path, video, self.video_folder),
                                                             limit_frames=1 if self.initial_frame == 0 else self.initial_frame,
                                                             offload_video_to_cpu=offload_cpu)
        # Times for each step:
        self.times = {
            "frame-load": 0,
            "det": 0,
            "init": 0,
            "match": 0,
        }

    def recover_tracks(self, unassigned_stracks, prev_frame_id, det_frame, previous_frame, current_frame,
                       activated_stracks, min_track_frames):
        """Decides which of the unmatched tracks are recovered in the current frame.

        Parameters
        ----------
            - unassigned_stracks: list of tracks left unmatched by the association step.
            - prev_frame_id: index of the previous frame, starting at zero.
            - det_frame: image of the detection frame (t_last - 1).
            - previous_frame: image of the previous frame.
            - current_frame: image of the current frame.
            - activated_stracks: list of tracks matched in the current frame.
            - min_track_frames: number of frames a track has to be maintained
              before it becomes eligible for recovery (F).

        Return
        ------
            - One entry per unmatched track, stating whether it is recovered and,
              if so, the bounding box and the mask obtained for the current frame.
        """
        # Reset SAM 2 state and free memory:
        self.sam2_predictor.reset_state(self.sam2_state)
        gc.collect()
        torch.cuda.empty_cache()
        new_round_info = []

        start_frame_load_time = time.time()

        adapt_prev_frame_id = prev_frame_id + self.initial_frame
        self.sam2_state = self.sam2_predictor.add_frames_to_state(self.sam2_state, os.path.join(self.data_path, self.video, self.video_folder),
                                                                  last_frame_to_add_idx=adapt_prev_frame_id + 2, offload_video_to_cpu=self.offload_cpu)

        end_frame_load_time = time.time()
        self.times["frame-load"] += end_frame_load_time - start_frame_load_time

        if len(unassigned_stracks) == 0:
            return []

        if len(activated_stracks) > 0:
            # Check overlapping between unassigned tracks and activated_stracks.
            ious = 1 - matching.iou_distance(unassigned_stracks, activated_stracks)
            # Get maximum IoU value per row:
            max_ious = ious.max(axis=1)

        start_det_time = time.time()

        # Get bounding boxes that overlap with the activated tracks -- the neighbourhood:
        bboxes_unassigned = [track.prev_tlwh for track in unassigned_stracks]
        bboxes_activated = [track.prev_tlwh for track in activated_stracks]
        idxs_overlap = self.check_overlap(bboxes_unassigned, bboxes_activated)
        bboxes_overlap = [bboxes_activated[i] for i in idxs_overlap]

        # Get the same bboxes for the detection frame. If we do not have det embeddings
        # established, we will do it:
        unassigned_tracks_no_det = [track for track in unassigned_stracks if track.det_frame == -1]
        bboxes_unassigned_det = [track.tm2_prev_tlwh for track in unassigned_tracks_no_det]
        bboxes_activated_det = [track.tm2_prev_tlwh for track in activated_stracks]
        idxs_overlap_det = self.check_overlap(bboxes_unassigned_det, bboxes_activated_det)
        bboxes_overlap_det = [bboxes_activated_det[i] for i in idxs_overlap_det]

        # Flag to indicate if at least one object is kept:
        at_least_one_obj = False

        # Masks in the detection frame (t_last - 1), computed once per track:
        if det_frame is not None:
            self.sam2_img_predictor.set_image(det_frame)
            self.masks_detection_frame(bboxes_unassigned_det, bboxes_overlap_det, unassigned_tracks_no_det,
                                       adapt_prev_frame_id - 1, det_frame)
            self.sam2_predictor.reset_state(self.sam2_state)

        end_det_time = time.time()
        self.times["det"] += end_det_time - start_det_time

        start_init_time = time.time()

        # Use SAM 2 video predictor to predict FIRST all masks from all objects in overlap (from 1st/2nd rounds):
        masks_overlap = self.compute_masks_overlap(bboxes_overlap, adapt_prev_frame_id, previous_frame)
        # Reset state of the predictor --- we will set the predicted masks directly but we will add unassigned tracks bounding boxes as well:
        self.sam2_predictor.reset_state(self.sam2_state)

        for i, mask in enumerate(masks_overlap["prev_frame"]):
            self.sam2_predictor.add_new_mask(self.sam2_state, adapt_prev_frame_id, i + len(unassigned_stracks), mask)

        # Append information from unassigned tracks:
        for i, track in enumerate(unassigned_stracks):
            # If the number of tracked frames is not greather than the threshold, we won't use this new round:
            if track.num_frames_tracked < min_track_frames:
                new_round_info.append({
                    "track": i,
                    "tracked": False,
                    "new_det": None,
                    "mask": None,
                    "lost_mask": None,
                    "prob": 0.0,
                    "reason": "NEF",
                })  # Not enough frames
                continue
            # Get the bounding box of the track and check if it is outside the frame:
            if (track.prev_tlwh[0] < 0 or track.prev_tlwh[1] < 0 or
                    track.prev_tlwh[0] + track.prev_tlwh[2] > previous_frame.shape[1] or
                    track.prev_tlwh[1] + track.prev_tlwh[3] > previous_frame.shape[0]):
                new_round_info.append({
                    "track": i,
                    "tracked": False,
                    "new_det": None,
                    "mask": None,
                    "lost_mask": None,
                    "prob": 0.0,
                    "reason": "OOB",
                })  # Out of bounds
                continue
            # If the max iou of the track with other first/second round tracks is under 0.1, we discard it:
            if len(activated_stracks) > 0 and max_ious[i] < MIN_OVERLAP:
                new_round_info.append({
                    "track": i,
                    "tracked": False,
                    "new_det": None,
                    "mask": None,
                    "lost_mask": None,
                    "prob": 0.0,
                    "reason": "ALN",
                })  # Alone track
                continue
            at_least_one_obj = True
            # If mask is not none, we can use it:
            if track.mask is not None:
                self.sam2_predictor.add_new_mask(self.sam2_state, adapt_prev_frame_id, i, track.mask)
            else:
                # Use the bounding box - adapted to tlbr. Control that coordinates are inside frame coordinates:
                prev_bbox_tlbr = [
                    int(max(0, track.prev_tlwh[0])),
                    int(max(0, track.prev_tlwh[1])),
                    int(min(previous_frame.shape[1] - 1, track.prev_tlwh[0] + track.prev_tlwh[2])),
                    int(min(previous_frame.shape[0] - 1, track.prev_tlwh[1] + track.prev_tlwh[3]))
                ]
                self.sam2_predictor.add_new_points_or_box(self.sam2_state, frame_idx=adapt_prev_frame_id, obj_id=i, box=prev_bbox_tlbr)

        # If we don't have at least one object to track, we will not continue tracking
        if not at_least_one_obj:
            return new_round_info

        end_init_time = time.time()
        self.times["init"] += end_init_time - start_init_time

        start_match_time = time.time()

        prev_masks = [None] * len(unassigned_stracks)

        for frame_id, object_ids, masks in self.sam2_predictor.propagate_in_video(self.sam2_state, start_frame_idx=adapt_prev_frame_id,
                                                                                  max_frame_num_to_track=1):
            for i, obj_id in enumerate(object_ids):
                # We are only interested in the unassigned tracks masks:
                if obj_id < len(unassigned_stracks):
                    if frame_id == adapt_prev_frame_id:
                        prev_mask = masks[i][0].cpu().numpy()
                        prev_masks[obj_id] = (prev_mask > 0).astype(np.uint8)
                    else:
                        # Get the mask:
                        mask = masks[i][0].cpu().numpy()
                        # Binarize with threshold at 0:
                        mask = (mask > 0).astype(np.uint8)

                        # Ensure we have both masks:
                        if prev_masks[obj_id].sum() > 0 and mask.sum() > 0:
                            # Extract embeddings for previous frame mask and current frame mask:
                            self.sam2_img_predictor.set_image(previous_frame)
                            prev_mask_embds, _ = extract_embeddings(prev_masks[obj_id], self.sam2_img_predictor, previous_frame)
                            prev_bbox = convert_to_tlbr(unassigned_stracks[obj_id].prev_tlwh)  # TLBR!
                            self.sam2_img_predictor.set_image(current_frame)
                            mask_embds, _ = extract_embeddings(mask, self.sam2_img_predictor, current_frame)
                            curr_bbox_est = convert_to_tlbr(unassigned_stracks[obj_id].tlwh)  # TLBR!

                            # Mask of the detection frame, computed the first time the track entered ORP:
                            det_embds = unassigned_stracks[obj_id].det_embedding
                            det_frame_id = unassigned_stracks[obj_id].det_frame
                            det_bbox = unassigned_stracks[obj_id].det_bbox
                            if det_embds is None:
                                # No embeddings available -> Discard this track:
                                new_round_info.append({
                                    "track": obj_id,
                                    "tracked": False,
                                    "new_det": None,
                                    "mask": None,
                                    "lost_mask": mask,
                                    "prob": 0.0,
                                    "reason": "NoDet",
                                })
                                continue

                            # Send to cuda all embeddings:
                            det_embds = torch.tensor(det_embds).to(torch.float32).cuda()
                            prev_mask_embds = torch.tensor(prev_mask_embds).to(torch.float32).cuda()
                            mask_embds = torch.tensor(mask_embds).to(torch.float32).cuda()
                            det_bbox = torch.tensor(det_bbox).to(torch.long).cuda()
                            prev_bbox = torch.tensor(prev_bbox).to(torch.long).cuda()
                            curr_bbox_est = torch.tensor(curr_bbox_est).to(torch.long).cuda()

                            mask_embds = torch.cat((det_embds, prev_mask_embds, mask_embds), dim=0).unsqueeze(0)
                            bboxes = torch.stack((det_bbox, prev_bbox, curr_bbox_est), dim=0).unsqueeze(0)

                            det_frame_tens = torch.tensor([det_frame_id]).unsqueeze(0).to(torch.long).cuda()
                            frame_id_tens = torch.tensor([prev_frame_id + 1]).unsqueeze(0).to(torch.long).cuda()

                            # Execute MIR:
                            output = self.mir(mask_embds, bboxes, det_frame_tens, frame_id_tens)

                            # Here we will check output value --> Decide the action:
                            if output >= self.gamma:
                                # New bounding box will be created from the displacement of the mask:
                                disp_box = self.box_from_mask_displacement(prev_masks[obj_id], mask, unassigned_stracks[obj_id])
                                new_round_info.append({
                                    "track": obj_id,
                                    "tracked": True,
                                    "new_det": disp_box,
                                    "mask": mask,
                                    "prob": output.item(),
                                })
                            else:
                                new_round_info.append({
                                    "track": obj_id,
                                    "tracked": False,
                                    "new_det": None,
                                    "mask": None,
                                    "lost_mask": mask,
                                    "prob": output.item(),
                                    "reason": "Filt",
                                })
                        else:
                            new_round_info.append({
                                "track": obj_id,
                                "tracked": False,
                                "new_det": None,
                                "mask": None,
                                "lost_mask": mask,
                                "prob": 0.0,
                                "reason": "NoMask",
                            })

        end_match_time = time.time()
        self.times["match"] += end_match_time - start_match_time

        return new_round_info

    def compute_masks_overlap(self, bboxes, prev_frame_id, previous_frame):
        """Function that computes the masks of the bounding boxes that overlap with the activated tracks.

        Parameters
        ----------
            - bboxes: list of bounding boxes.
            - prev_frame_id: frame id to use.
            - previous_frame: image of the previous frame.

        Return
        ------
            - A list of masks.
        """
        predicted_masks = {
            "prev_frame": [],
            "current_frame": []
        }
        if len(bboxes) == 0:
            # No bboxes to track:
            return predicted_masks
        for i, bbox in enumerate(bboxes):
            bbox_tlbr = [
                int(max(0, bbox[0])),
                int(max(0, bbox[1])),
                int(min(previous_frame.shape[1] - 1, bbox[0] + bbox[2])),
                int(min(previous_frame.shape[0] - 1, bbox[1] + bbox[3]))
            ]
            # Append the bbox to video predictor in previous frame:
            self.sam2_predictor.add_new_points_or_box(self.sam2_state, frame_idx=prev_frame_id, obj_id=i, box=bbox_tlbr)
        # Predict the masks:
        for frame_id, object_ids, masks in self.sam2_predictor.propagate_in_video(self.sam2_state, start_frame_idx=prev_frame_id,
                                                                                  max_frame_num_to_track=1):
            for i, obj_id in enumerate(object_ids):
                if obj_id < len(bboxes):
                    mask = masks[i][0].cpu().numpy()
                    # Binarize with threshold at 0:
                    mask = (mask > 0).astype(np.uint8)
                    if frame_id == prev_frame_id:
                        predicted_masks["prev_frame"].append(mask)
                    else:
                        predicted_masks["current_frame"].append(mask)

        return predicted_masks

    def masks_detection_frame(self, bboxes_unassigned, bboxes_overlap, tracks_list, track_frame_id, frame):
        """Function that computes, in the detection frame, the masks of the tracks that
        enter ORP for the first time, and stores their embeddings in the tracks.

        Parameters
        ----------
            - bboxes_unassigned: bounding boxes of the tracks without detection frame.
            - bboxes_overlap: bounding boxes of their neighbouring tracks.
            - tracks_list: tracks the bounding boxes belong to.
            - track_frame_id: index of the detection frame.
            - frame: image of the detection frame.
        """
        # If there are no bboxes with value different to None we will return:
        if len([bbox for bbox in bboxes_unassigned if bbox is not None]) == 0 and len([bbox for bbox in bboxes_overlap if bbox is not None]) == 0:
            return
        if len(bboxes_unassigned) == 0:
            # No bboxes to track:
            return
        for i, bbox in enumerate(bboxes_unassigned):
            if bbox is None:
                continue
            bbox_tlbr = [
                int(max(0, bbox[0])),
                int(max(0, bbox[1])),
                int(min(frame.shape[1] - 1, bbox[0] + bbox[2])),
                int(min(frame.shape[0] - 1, bbox[1] + bbox[3]))
            ]
            # Append the bbox to video predictor in the detection frame:
            self.sam2_predictor.add_new_points_or_box(self.sam2_state, frame_idx=track_frame_id, obj_id=i, box=bbox_tlbr)
        for i, bbox in enumerate(bboxes_overlap):
            if bbox is None:
                continue
            bbox_tlbr = [
                int(max(0, bbox[0])),
                int(max(0, bbox[1])),
                int(min(frame.shape[1] - 1, bbox[0] + bbox[2])),
                int(min(frame.shape[0] - 1, bbox[1] + bbox[3]))
            ]
            self.sam2_predictor.add_new_points_or_box(self.sam2_state, frame_idx=track_frame_id, obj_id=i + len(bboxes_unassigned), box=bbox_tlbr)
        # Predict the masks:
        for frame_id, object_ids, masks in self.sam2_predictor.propagate_in_video(self.sam2_state, start_frame_idx=track_frame_id,
                                                                                  max_frame_num_to_track=1):
            for i, obj_id in enumerate(object_ids):
                # We are only interested in the unassigned tracks masks:
                if obj_id < len(bboxes_unassigned):
                    mask = masks[i][0].cpu().numpy()
                    # Binarize with threshold at 0:
                    mask = (mask > 0).astype(np.uint8)
                    # Get embeddings:
                    mask_embds, _ = extract_embeddings(mask, self.sam2_img_predictor, frame)
                    if frame_id == track_frame_id:
                        tracks_list[obj_id].det_embedding = mask_embds
                        tracks_list[obj_id].det_frame = track_frame_id
                        tracks_list[obj_id].det_bbox = convert_to_tlbr(bboxes_unassigned[obj_id])

    def box_from_mask_displacement(self, prev_mask, actual_mask, track):
        """Function that extracts the detection bounding box from the previous
        and the actual obtained mask, in order to be posteriorly assigned to the
        track information.

        Parameters
        ----------
            - prev_mask: mask obtained for the previous frame for the tracked object.
            - actual_mask: mask that was obtained for the actual frame for the tracked object.
            - track: information from the tracked object.

        Return
        ------
            - Coordinates [top, left, width, height] of the displaced bounding box.
        """
        prev_mask_true_idx = np.argwhere(prev_mask)

        # Minmax of the mask - bounding box:
        prev_mask_y_min, prev_mask_x_min = prev_mask_true_idx.min(axis=0)
        prev_mask_y_max, prev_mask_x_max = prev_mask_true_idx.max(axis=0)
        prev_mask_y_delta, prev_mask_x_delta = prev_mask_y_max - prev_mask_y_min, prev_mask_x_max - prev_mask_x_min

        prev_mask_center = [prev_mask_x_min + prev_mask_x_delta / 2, prev_mask_y_min + prev_mask_y_delta / 2]

        actual_mask_true_idx = np.argwhere(actual_mask)

        # Minmax of the mask - bounding box:
        actual_mask_y_min, actual_mask_x_min = actual_mask_true_idx.min(axis=0)
        actual_mask_y_max, actual_mask_x_max = actual_mask_true_idx.max(axis=0)
        actual_mask_y_delta, actual_mask_x_delta = actual_mask_y_max - actual_mask_y_min, actual_mask_x_max - actual_mask_x_min

        actual_mask_center = [actual_mask_x_min + actual_mask_x_delta / 2, actual_mask_y_min + actual_mask_y_delta / 2]

        # Displacement:
        disp = [actual_mask_center[0] - prev_mask_center[0], actual_mask_center[1] - prev_mask_center[1]]

        # Compute displaced box:
        disp_box = [track.prev_tlwh[0] + disp[0], track.prev_tlwh[1] + disp[1], track.prev_tlwh[2], track.prev_tlwh[3]]

        return disp_box

    def check_overlap(self, bboxes_unassigned, bboxes_activated):
        """Function that checks the overlap between the bounding boxes of the unassigned
        tracks and the activated ones.

        Parameters
        ----------
            - bboxes_unassigned: list of bounding boxes of the unassigned tracks.
            - bboxes_activated: list of bounding boxes of the activated tracks.

        Return
        ------
            - A list of indexes of the activated tracks that overlap with the unassigned ones.
        """
        bboxes_overlap = []
        for i, bbox_activated in enumerate(bboxes_activated):
            for bbox_unassigned in bboxes_unassigned:
                if bbox_activated is None or bbox_unassigned is None:
                    continue  # IoA will be 0 - no overlap
                if self.compute_ioa(bbox_unassigned, bbox_activated) > MIN_OVERLAP:
                    bboxes_overlap.append(i)
                    break
        return bboxes_overlap

    def compute_ioa(self, bbox1, bbox2):
        """Function that computes the Intersection over Area (IoA) between two bounding boxes.
        Intersection over Area = Intersection Area / min(Area of bbox1, Area of bbox2).

        Parameters
        ----------
            - bbox1: first bounding box.
            - bbox2: second bounding box.

        Return
        ------
            - IoA value.
        """
        # Get the coordinates of the intersection rectangle:
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        # If the boxes do not overlap, return 0:
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Compute the area of the intersection rectangle:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Compute the area of both bounding boxes:
        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]

        # Compute the IoA:
        return intersection_area / min(bbox1_area, bbox2_area)


def convert_to_tlbr(tlwh):
    """Function that converts a bounding box from top-left width-height to top-left bottom-right format.

    Parameters
    ----------
        - tlwh: bounding box in top-left width-height format.

    Return
    ------
        - Bounding box in top-left bottom-right format.
    """
    return [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

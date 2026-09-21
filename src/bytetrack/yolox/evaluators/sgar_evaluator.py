"""Evaluation loop for SGAR-MOT.

Adapted from ByteTrack's MOTEvaluator: one tracker per sequence, results written in MOTChallenge format, one .txt per sequence. 
"""

import contextlib
import io
import itertools
import json
import os
import tempfile
import time
from collections import defaultdict

import cv2
import torch
from loguru import logger
from tqdm import tqdm

from yolox.tracker.sgar_tracker import SGARTracker
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class SGAREvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        video_name = None

        comp_times_per_frame = {
            'detection': [],
            'tracking_full': [],
            'tracking-1st': [],
            'tracking-2nd': [],
            'tracking-orp': [],
            'tracking-orp-frame-load': [],
            'tracking-orp-det': [],
            'tracking-orp-init': [],
            'tracking-orp-match': [],
        }

        num_total_objects = 0

        tracker = SGARTracker(self.args)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()

                ## Removed ByteTrack per-video tuning tricks.

                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                # If in result_folder there is a save file with video name, we skip it directly:
                if os.path.exists(f"{result_folder}/{video_name}.txt"):
                    continue

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    # ORP holds the SAM 2 state of the sequence, so the tracker is rebuilt per video:
                    tracker = SGARTracker(self.args, video_name)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                start_time_detection = time.time()

                outputs = model(imgs)
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

                end_time_detection = time.time()
                comp_times_per_frame['detection'].append(end_time_detection - start_time_detection)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            start_time_tracking = time.time()

            # run tracking
            if outputs[0] is not None:
                imgs_np = cv2.imread(self.args.data_path + img_file_name[0])
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, imgs_np)
                # Get counter times:
                for key in tracker.comp_times:
                    comp_times_per_frame[key].append(tracker.comp_times[key])

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        num_total_objects += 1
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
            else:
                # Only increase frame id for consistency:
                tracker.frame_id += 1

            end_time_tracking = time.time()
            comp_times_per_frame['tracking_full'].append(end_time_tracking - start_time_tracking)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        # Show computation times average:
        for key, value in comp_times_per_frame.items():
            if len(value) > 0:
                logger.info(f"Average {key} time: {sum(value) / len(value)}")

        logger.info(f"Total objects tracked: {num_total_objects}")

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_fromdetector(
        self,
        model,
        distributed=False,
        half=False,
        result_folder=None
    ):
        """
        Same evaluation loop, taking the detections from a folder of files instead
        of running the detector.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor

        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        video_name = None

        comp_times_per_frame = {
            'detection': [],
            'tracking_full': [],
            'tracking-1st': [],
            'tracking-2nd': [],
            'tracking-orp': [],
            'tracking-orp-frame-load': [],
            'tracking-orp-det': [],
            'tracking-orp-init': [],
            'tracking-orp-match': [],
        }

        num_total_objects = 0

        tracker = SGARTracker(self.args)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()

                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                # If in result_folder there is a save file with video name, we skip it directly:
                if os.path.exists(f"{result_folder}/{video_name}.txt"):
                    continue

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    # ORP holds the SAM 2 state of the sequence, so the tracker is rebuilt per video:
                    tracker = SGARTracker(self.args, video_name)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                start_time_detection = time.time()

                outputs = self.get_detections_from_file(video_name, frame_id, self.args.det_folder)

                end_time_detection = time.time()
                comp_times_per_frame['detection'].append(end_time_detection - start_time_detection)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            start_time_tracking = time.time()

            # run tracking
            if outputs[0] is not None:
                imgs_np = cv2.imread(self.args.data_path + img_file_name[0])
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, imgs_np)
                # Get counter times:
                for key in tracker.comp_times:
                    comp_times_per_frame[key].append(tracker.comp_times[key])

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        num_total_objects += 1
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
            else:
                # Only increase frame id for consistency:
                tracker.frame_id += 1

            end_time_tracking = time.time()
            comp_times_per_frame['tracking_full'].append(end_time_tracking - start_time_tracking)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        # Show computation times average:
        for key, value in comp_times_per_frame.items():
            if len(value) > 0:
                logger.info(f"Average {key} time: {sum(value) / len(value)}")

        logger.info(f"Total objects tracked: {num_total_objects}")

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def get_detections_from_file(self, video_name, frame_id, detection_folder):
        """
        Get detections from file for a specific video and frame.

        Args:
            video_name (str): Name of the video.
            frame_id (int): Frame index.
            detection_folder (str): Path to the folder containing detection files.

        Returns:
            outputs (list): List of detections for the specified frame.
        """
        # Path to detection file (json)
        detection_file = os.path.join(detection_folder, f"{video_name}.json")
        if not os.path.exists(detection_file):
            raise FileNotFoundError(f"Detection file {detection_file} does not exist.")

        # Load detections from the file:
        with open(detection_file, 'r') as f:
            detections = json.load(f)
        # Get unique list of frame_ids:
        frame_ids = set(det['image_id'] for det in detections)
        # Order the frame_ids:
        frame_ids = sorted(frame_ids)
        # Filter detections for the specific frame_id:
        detections_frame = [
            det["bbox"] + [det["score"]] + [det["category_id"]] + [0] for det in detections if det['image_id'] == frame_ids[frame_id - 1] and det["score"] >= self.confthre
        ]

        # Convert to tensor and send to cuda:
        if len(detections_frame) == 0:
            return [None]
        outputs = torch.tensor(detections_frame, dtype=torch.float32).cuda()
        return [outputs]

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

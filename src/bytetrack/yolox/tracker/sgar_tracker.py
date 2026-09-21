"""SGAR-MOT: ByteTrack with the Object Recovery Protocol.

The tracker is ByteTrack (based on byte_tracker.py) -- same association strategy and basic configuration, adding the implementation of our ORP.
"""

import time

import numpy as np

from yolox.tracker import matching

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter

from sgar_mot.orp import ORP


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, mask=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        ### [SGAR-MOT] Attributes needed by ORP:
        self.mask = mask
        self.prev_tlwh = None  # Previous bounding box (if exists).
        self.tm2_prev_tlwh = None  # Bounding box from two frames away (if exists).
        self.num_frames_tracked = 0

        # State of the detection frame (t_last - 1), computed once per ORP entry:
        self.det_embedding = None
        self.det_bbox = None
        self.det_frame = -1

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                # [SGAR-MOT] Snapshot the position of the track in t-1 and t-2, which
                # ORP uses to prompt SAM 2.
                st.tm2_prev_tlwh = st.prev_tlwh.copy() if st.prev_tlwh is not None else None
                st.prev_tlwh = st.tlwh.copy()
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        # [SGAR-MOT] Update mask. If None, it will be removed (1st and 2nd associations):
        self.mask = new_track.mask
        # If reactivated, number of tracked frames will be restored to 1:
        self.num_frames_tracked = 1

        self.det_embedding = None
        self.det_bbox = None
        self.det_frame = -1

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # [SGAR-MOT] Update mask. If None, it will be removed (1st and 2nd associations):
        self.mask = new_track.mask
        # Increase number of tracked frames:
        self.num_frames_tracked += 1

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class SGARTracker(object):
    def __init__(self, args, video=None, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # [SGAR-MOT] Object Recovery Protocol, built per sequence:
        if self.args.orp:
            self.orp = ORP(self.args.sam_checkpoint, self.args.sam_config, self.args.mir_config,
                           self.args.data_path, video,
                           val_half="train" in args.data_path and "mot" in args.data_path,
                           offload_cpu=self.args.offload_cpu, gamma=self.args.gamma)
        else:
            self.orp = None
            print("ORP disabled: running the baseline tracker.")

        self.frame_history = []

        self.comp_times = {
            'tracking-1st': 0,
            'tracking-2nd': 0,
            'tracking-orp': 0,
            'tracking-orp-frame-load': 0,
            'tracking-orp-det': 0,
            'tracking-orp-init': 0,
            'tracking-orp-match': 0,
        }

    def update(self, output_results, img_info, img_size, curr_frame):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # [SGAR-MOT] ORP reasons over the previous frames, so they are kept:
        self.frame_history.append(curr_frame)

        if self.args.det_folder is None:
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                output_results = output_results.cpu().numpy()
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale
        else:
            # Detections read from file: they are already in image coordinates.
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]  # x1y1x2y2

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        start_time_1st = time.time()

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            # [SGAR-MOT] If we had saved det frame, we will remove it:
            track.det_frame = -1
            track.det_embedding = None
            track.det_bbox = None

        self.comp_times['tracking-1st'] += time.time() - start_time_1st

        start_time_2nd = time.time()

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            # [SGAR-MOT] If we had saved det frame, we will remove it:
            track.det_frame = -1
            track.det_embedding = None
            track.det_bbox = None

        self.comp_times['tracking-2nd'] += time.time() - start_time_2nd

        start_time_orp = time.time()

        ############################################################################################################
        # [SGAR-MOT] Step 3.1: Object Recovery Protocol over the tracks left unmatched.
        if self.orp is not None:
            unassigned_stracks = [r_tracked_stracks[it] for it in u_track]
            # Initially, all elements from unassigned_stracks would be lost in theory:
            u_track = list(range(len(unassigned_stracks)))

            # Previous and t-2 frames -- the latter is the detection frame:
            prev_frame = self.frame_history[-2] if len(self.frame_history) > 1 else None
            det_frame = self.frame_history[-3] if len(self.frame_history) > 2 else None

            # Frame ID - previous frame ID and starting at zero (so we substract two to current frame ID value):
            orp_info = self.orp.recover_tracks(unassigned_stracks, self.frame_id - 2, det_frame, prev_frame, curr_frame,
                                               activated_starcks, self.args.min_track_frames)

            self.comp_times['tracking-orp-frame-load'] = self.orp.times['frame-load']
            self.comp_times['tracking-orp-det'] = self.orp.times['det']
            self.comp_times['tracking-orp-init'] = self.orp.times['init']
            self.comp_times['tracking-orp-match'] = self.orp.times['match']

            orp_utrack = []
            for track_info in orp_info:
                track = unassigned_stracks[track_info["track"]]
                # If the track is not recovered, we append it to u_track:
                if not track_info["tracked"]:
                    orp_utrack.append(track_info["track"])
                else:
                    # Define STrack class with the recovered detection:
                    new_track = STrack(track_info["new_det"], track.score, track_info["mask"])
                    # Control update:
                    if track.state == TrackState.Tracked:
                        track.update(new_track, self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(new_track, self.frame_id, new_id=False)
                        refind_stracks.append(track)

            # Update unmatched tracks -- this is the last association step:
            u_track = orp_utrack
            # Mark them as lost:
            for it in u_track:
                track = unassigned_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        ############################################################################################################

        self.comp_times['tracking-orp'] += time.time() - start_time_orp

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

"""Spatiotemporal positional encoding of MIR.

Each mask token is encoded with three magnitudes measured with respect to the
anchor (the last observation matched to a detection): the temporal distance
(E_t), the relative scale (E_s) and the relative displacement (E_d).
"""

import torch
from torch import nn

from .positional_encodings.torch_encodings import PositionalEncoding3D


class SpatioTemporalPositionalEncoding(nn.Module):
    """Positional encoding over (displacement, scale, time) w.r.t. the anchor mask."""

    def __init__(self, d_model, dropout=0.1, max_temp_dist=30, max_distance_dist=105, max_size_dist=105,
                 batch_first=False, device='cuda'):
        super(SpatioTemporalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        self.device = device

        self.max_distance_dist = max_distance_dist
        self.max_size_dist = max_size_dist
        self.max_temp_dist = max_temp_dist

        distance_range = self.max_distance_dist * 2 + 1
        size_range = self.max_size_dist * 2 + 1  # Smaller and bigger
        temp_range = self.max_temp_dist * 2 + 1  # Previous and forward frames

        p_enc_3d = PositionalEncoding3D(channels=self.d_model)  # Will receive a 5d tensor of size (batch_size, x, y, z, ch)
        pe = p_enc_3d(torch.zeros(1, distance_range, size_range, temp_range, d_model))  # Will return a 5d tensor of size (batch_size, x, y, z, ch)
        pe = pe.squeeze(0)  # Remove the batch dimension
        pe = pe.to(torch.float16)  # We use float16 to save memory
        self.pe = pe

        # We remove unnecerary stuff that may cause memory leaks
        del p_enc_3d.cached_penc
        del p_enc_3d.inv_freq

        assert self.batch_first == True, "batch_first must be True"

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, msk, msk_bboxes, num_masks, det_frame_id, t_frame_id):
        ref_bbox = msk_bboxes[:, -1:, :].clone()  # [batch_size, 4]  | Our reference bbox is the last one in the memory

        if not self.batch_first:
            raise NotImplementedError("Not implemented for batch_first=False")

        # We first add "fake" bbox coordinates to the extra token (DEC).
        msk_bboxes = self._insert_fake_bboxes(msk=msk, msk_bboxes=msk_bboxes, ref_bbox=ref_bbox, num_masks=num_masks)

        # Get the temporal indices for the embeddings
        msk_t_inds = self._get_temporal_ids(msk=msk, num_masks=num_masks, det_frame_id=det_frame_id, t_frame_id=t_frame_id)

        # Get the spatial indices for the embeddings
        msk_xy_inds, msk_size_inds = self._get_spatial_ids(msk_bboxes=msk_bboxes)

        # Retrieve the encodings for the masks
        msk_indices = zip(msk_xy_inds, msk_size_inds, msk_t_inds)
        msk_encodings = []
        for it, sample_ind in enumerate(msk_indices):  # We iterate batch_size elems
            elem_msk_indices = zip(sample_ind[0], sample_ind[1], sample_ind[2])  # i[0], i[1], i[2]
            elem_msk_encodings = torch.stack([self.pe[i[0].item(), i[1].item(), i[2].item()] for i in elem_msk_indices])  # Size [num_msks, d_model]
            msk_encodings.append(elem_msk_encodings)

        msk_encodings = torch.stack(msk_encodings)  # Size [batch_size, num_msks, d_model]
        msk_encodings = msk_encodings.to(self.device)

        x = msk + msk_encodings

        return self.dropout(x)

    def _insert_fake_bboxes(self, msk, msk_bboxes, ref_bbox, num_masks):
        """We need to insert a fake bbox for the [DEC] token (as it does not have one).
        It takes the last bbox in the memory (ref_bbox). num_masks is the number of tokens
        in the input, this is, the three masks plus the [DEC] token."""
        num_additional_candidates = 1  # We do not have non-token in this setup.

        assert msk.shape[1] == (msk_bboxes.shape[1] + num_additional_candidates), "The number of tokens in the candidates is not correct"
        assert msk_bboxes.shape[1] == num_masks - num_additional_candidates, "The number of bboxes in the candidates is not correct"

        can_with_pads = [torch.cat([msk_bboxes[:, [i], :]], dim=1) for i in range(msk_bboxes.shape[1])]
        zero_bboxes = [ref_bbox]
        fake_bboxes = torch.cat(can_with_pads + zero_bboxes, dim=1)

        return fake_bboxes

    def _get_temporal_ids(self, msk, num_masks, det_frame_id, t_frame_id):
        """Temporal index of every token, taking the detection frame as origin:
        DET => 0, T-1 => Dist(T-1 - DET), T => Dist(T - DET). Input is supposed to
        go in that order.
        """
        if not self.batch_first:
            raise NotImplementedError("Not implemented for batch_first=False")

        dist_det_t = t_frame_id - det_frame_id

        msk_len = msk.shape[1]
        assert msk_len % num_masks == 0, "The number of tokens in the candidates is not correct"

        # These are the temporal IDs that we are going to use:
        # [MASK_DET] [MASK_T-1] [MASK_T] [DEC_TOKEN]
        temporal_ids = torch.Tensor([[0, d_d_t - 1, d_d_t, 0] for d_d_t in dist_det_t]).to(torch.long)
        # Output token will be given 0 as temporal distance
        # Limit max_temp_dist:
        temporal_ids = torch.clamp(temporal_ids, min=-self.max_temp_dist, max=self.max_temp_dist).to(torch.long)

        return temporal_ids

    def _get_spatial_ids(self, msk_bboxes, range_factor=15.0):
        if not self.batch_first:
            raise NotImplementedError("Not implemented for batch_first=False")

        batch_size = msk_bboxes.shape[0]

        ref_bbox = msk_bboxes[:, -1:, :]  # [batch_size, 4]  | Our reference bbox is the last one in the memory

        # We will replicate the reference bbox to the same size as the can_bboxes and the whole memory
        ref_bbox_msk = ref_bbox.repeat(1, msk_bboxes.shape[1], 1)  # [batch_size, num_masks, 4]

        # Get the spatial indices for the embeddings. We will flatten the bboxes
        # along the batch dimension: [batch_size, num_masks, 4] -> [batch_size * num_masks, 4]
        msk_bboxes_flattened = msk_bboxes.view(-1, 4)
        ref_bbox_msk_flattened = ref_bbox_msk.view(-1, 4)

        msk_xy_distance_flattened, msk_size_distance_flattened = self.extract_distance_values(bbox=msk_bboxes_flattened, ref_bbox=ref_bbox_msk_flattened)
        msk_xy_distance = msk_xy_distance_flattened.view(batch_size, -1)

        msk_size_distance = msk_size_distance_flattened.view(batch_size, -1)

        # Values roughly go from -7 to 7. Much like MEGA does, we multiply by a
        # factor (15, instead of 100) to get a wider range of values from -105 to 105
        msk_xy_distance = torch.clamp(msk_xy_distance * range_factor, min=-self.max_distance_dist, max=self.max_distance_dist).to(torch.long)
        msk_size_distance = torch.clamp(msk_size_distance * range_factor, min=-self.max_size_dist, max=self.max_size_dist).to(torch.long)
        msk_size_distance = torch.clamp(msk_size_distance, min=-self.max_size_dist, max=self.max_size_dist).to(torch.long)  # I have to do it 2 times to avoid area-zero bboxes that overflow the metric

        msk_xy_inds = msk_xy_distance + self.max_distance_dist
        msk_size_inds = msk_size_distance + self.max_size_dist

        return msk_xy_inds, msk_size_inds

    @staticmethod
    def extract_distance_values(bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.tensor_split(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.tensor_split(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref
        delta_x = delta_x / bbox_width
        delta_x = torch.pow(delta_x, 2)

        delta_y = center_y - center_y_ref
        delta_y = delta_y / bbox_height
        delta_y = torch.pow(delta_y, 2)

        # We will use the euclidean distance, instead of element-by-element distance
        xy_distance = torch.sqrt(delta_x + delta_y)
        xy_distance = (xy_distance + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref
        delta_width = (delta_width + 1e-3).log()

        delta_height = bbox_height / bbox_height_ref
        delta_height = (delta_height + 1e-3).log()

        size_distance = delta_width + delta_height

        return xy_distance, size_distance

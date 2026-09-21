"""Mask preprocessing and embedding extraction utils for MIR.

A binary mask is padded to a square, resized to 256x256 and pushed through the
prompt encoder of SAM 2. The resulting dense embedding is added to the image
embedding of the frame (get_image_mask_embeddings, added to SAM 2 by
SGAR-MOT), and the region of the object is finally pooled with ROI Align, so
that every mask becomes a fixed-size token for MIR.
"""

import cv2
import numpy as np
import torch

from sgar_mot.mir.roi_align import ROIAlign


def extract_embeddings(mask, sam_predictor, image, bounding_box=None):
    """Extraction of the embeddings for a combination of image + mask.

    sam_predictor must already have been set on the image. If bounding_box is
    None, the minmax bounding box of the mask is used.
    """
    # Prepare segmentation mask - padding to largest size:
    original_height, original_width = mask.shape

    # First, we will take the largest size:
    max_side_size = max(original_height, original_width)
    new_mask_shape = (max_side_size, max_side_size)

    # Create the mask:
    square_mask = np.zeros(new_mask_shape, mask.dtype)

    # Padding computation:
    vertical_padding = max_side_size - original_height
    horizontal_padding = max_side_size - original_width

    # Put mask information. Only vertical or horizontal padding will be 0:
    if vertical_padding == 0:
        square_mask[:, :-horizontal_padding] = mask
    else:
        square_mask[:-vertical_padding, :] = mask

    # Resize to 256 x 256:
    resized_mask = cv2.resize(square_mask.astype(np.uint8), (256, 256))
    resized_mask = resized_mask.astype(np.int8)

    # Replace background zero values to a more negative value:
    resized_mask[resized_mask == 0] = -10

    # Extract mixed embeddings:
    img_mask_embeddings = sam_predictor.get_image_mask_embeddings(torch.tensor(resized_mask).unsqueeze(0).unsqueeze(0).float().cuda())

    # Create bounding box to use in ROI Align from segmentation mask if not provided.
    if bounding_box is None:
        bbox = minmax_bounding_box(mask)
    else:
        bbox = bounding_box

    # ROI Align only allows an scale factor for full image. Therefore, to handle with our case in which we have two
    # scale factors, we need to scale bounding box before providing it to the input:
    scale_w = image.shape[1] / img_mask_embeddings.shape[3]
    scale_h = image.shape[0] / img_mask_embeddings.shape[2]

    # New bounding box -- scaled:
    new_bbox = [bbox[0] / scale_w, bbox[1] / scale_h, bbox[2] / scale_w, bbox[3] / scale_h]

    # Extract ROI Align embeddings -- as bounding box has already been adapted to actual feature map, no scale is needed:
    roi_align = ROIAlign(output_size=(7, 7), spatial_scale=1, sampling_ratio=-1)
    # Processed bounding box, using necessary format:
    processed_bbox = torch.tensor(np.concatenate(([0], new_bbox))).unsqueeze(0).cuda()
    # Apply ROI Align:
    img_mask_embeddings = roi_align(img_mask_embeddings, processed_bbox)

    return img_mask_embeddings.detach().cpu().numpy(), bbox


def minmax_bounding_box(mask):
    """Extraction of the minmax bounding box of a mask."""
    # If empty mask, return 0 as bbox -- all coordinates to zero:
    if np.sum(mask) == 0:
        return [0, 0, 0, 0]

    mask_true_idx = np.argwhere(mask)

    # Minmax of the mask - bounding box:
    mask_y_min, mask_x_min = mask_true_idx.min(axis=0)
    mask_y_max, mask_x_max = mask_true_idx.max(axis=0)

    return [int(mask_x_min), int(mask_y_min), int(mask_x_max), int(mask_y_max)]


def bbox_as_mask(bbox, shape):
    """Function that creates a mask from a bounding box."""
    mask = np.zeros(shape, np.uint8)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    return mask

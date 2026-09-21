from torch import nn
from torchvision.ops import roi_align


class ROIAlign(nn.Module):
    """Class representing the execution of ROIAlign operation, based on
    https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        """Initialization method.

        Parameters
        ----------
            - output_size: expected output size for the feature map.
            - spatial_scale: how input bounding boxes should be scaled.
            - sampling_ratio: number of imput samples in the interpolation grid
              to be taken for each output sample.
        """

        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input_imgs, bboxes):
        """Forwarding method that runs ROI Align process.

        Parameters
        ----------
            - input_imgs: input images (N, C, H, W).
            - bboxes: regions of interest to be used (N, X1, Y1, X2, Y2).

        Return
        ------
            - ROI Align output from the torchvision function.
        """

        if input_imgs.is_quantized:
            input_imgs = input_imgs.dequantize()

        return roi_align(input_imgs, bboxes.to(input_imgs.dtype), self.output_size, self.spatial_scale, self.sampling_ratio,
                         aligned=False)

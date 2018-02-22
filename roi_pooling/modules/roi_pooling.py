"""TODO
"""
import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..functions.roi_pooling import roi_pooling_2d
from ..functions.roi_pooling import roi_pooling_2d_pytorch


class ROIPooling2d(nn.Module):
    """Spatial Region of Interest (ROI) pooling.

    This function acts similarly to :class:`~pytorch.nn.MaxPool2d`, but
    it computes the maximum of input spatial patch for each channel
    with the region of interest. This module only works with CUDA tensors.
    Take a look at the :class:`~ROIPooling2dPytorch` for an architecture
    agnostic implementation.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    Args:
        x (~pytorch.autograd.Variable): Input variable. The shape is expected
            to be 4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~pytorch.autograd.Variable): Input roi variable. The shape is
            expected to be (m: num-rois, 5), and each roi is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    Returns:
        `~pytorch.autograd.Variable`: Output variable.
    """

    def __init__(self, output_size, spatial_scale=1.0):
        super(ROIPooling2d, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pooling_2d(input, rois, self.output_size,
                              self.spatial_scale)

    def __repr__(self):
        return ('{}(output_size={}, spatial_scale={:.6f})'.format(
            self.__class__.__name__, str(self.output_size),
            str(self.spatial_scale)))


class ROIPooling2dPytorch(nn.Module):
    """Spatial Region of Interest (ROI) pooling.

    This function acts similarly to :class:`~ROIPooling2d`, but performs a
    python loop over ROI. Note that this is not a direct replacement of that
    operation and viceversa.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    Args:
        x (~pytorch.autograd.Variable): Input variable. The shape is expected
            to be 4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~pytorch.autograd.Variable): Input roi variable. The shape is
            expected to be (m: num-rois, 5), and each roi is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    Returns:
        `~pytorch.autograd.Variable`: Output variable.
    """

    def __init__(self, output_size, spatial_scale=1.0):
        super(ROIPooling2dPytorch, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pooling_2d_pytorch(input, rois, self.output_size,
                                      self.spatial_scale)

    def __repr__(self):
        return ('{}(output_size={}, spatial_scale={:.6f})'.format(
            self.__class__.__name__, str(self.output_size),
            str(self.spatial_scale)))

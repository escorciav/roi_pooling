"Test foward and backward for ROIPooling"
import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from roi_pooling.functions.roi_pooling import roi_pooling_2d
from roi_pooling.functions.roi_pooling import roi_pooling_2d_pytorch


class TestROIPooling2D(unittest.TestCase):
    "TODO"

    def setUp(self):
        self.batch_size = 3
        self.n_channels = 4
        self.input_size = (12, 8)
        self.output_size = (5, 7)
        self.spatial_scale = 0.6
        x_np = np.arange(self.batch_size * self.n_channels *
                         self.input_size[0] * self.input_size[1],
                         dtype=np.float32)
        x_np = x_np.reshape((self.batch_size, self.n_channels,
                             *self.input_size))
        np.random.shuffle(x_np)
        self.x = torch.from_numpy(2 * x_np / x_np.size - 1)
        self.rois = torch.FloatTensor([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]
        ])
        self.n_rois = self.rois.shape[0]

    def check_forward(self, x, rois):
        y_var = roi_pooling_2d(
            x, rois, self.output_size,
            spatial_scale=self.spatial_scale)
        self.assertIsInstance(y_var.data, torch.cuda.FloatTensor)

        d_output_shape = (self.n_rois, self.n_channels, *self.output_size)
        self.assertEqual(d_output_shape, tuple(y_var.data.size()))

    def check_forward_functional(self, x, rois):
        """crosscheck forward with naive roi_pooling"""
        # Set scale to 1.0 given that casting may be different
        prev_scale = self.spatial_scale
        self.spatial_scale = 1.0
        y_var = roi_pooling_2d(
            x, rois, self.output_size,
            spatial_scale=self.spatial_scale)
        d_y = roi_pooling_2d_pytorch(
            x, rois, self.output_size, self.spatial_scale)
        np.testing.assert_almost_equal(y_var.data.cpu().numpy(),
                                       d_y.data.cpu().numpy())
        self.spatial_scale = prev_scale

    def check_backward(self, x, rois):
        # gradchek takes a tuple of tensor as input, check if your gradient
        # evaluated with these tensors are close enough to numerical
        # approximations and returns True if they all verify this condition.
        input = (x, rois)
        self.assertTrue(
            gradcheck(roi_pooling_2d, input, eps=1e-6, atol=2e-2,
                      raise_exception=True))

    def test_forward_gpu(self):
        x = self.x.cuda()
        rois = self.rois.cuda()
        self.check_forward(x, rois)
        self.check_forward_functional(x, rois)

    def test_backward_gpu(self):
        x = self.x.cuda()
        x.requires_grad_(True)
        rois = self.rois.cuda()
        self.check_backward(x, rois)


if __name__ == '__main__':
    unittest.main()

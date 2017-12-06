import unittest

import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck

from roi_pooling import ROIPooling2d, roi_pooling_2d, roi_pooling_2d_naive


class TestROIPooling2D(unittest.TestCase):
    "TODO"

    def setUp(self):
        self.batch_size = 3
        self.n_channels = 4
        self.input_size = (12, 8)
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
        self.output_size = (5, 7)
        self.spatial_scale = 0.6
        self.n_rois = self.rois.shape[0]

    def check_forward(self, x_var, rois_var):
        y_var = roi_pooling_2d(
            x_var, rois_var, self.output_size,
            spatial_scale=self.spatial_scale)
        self.assertIsInstance(y_var.data, torch.cuda.FloatTensor)

        d_output_shape = (self.n_rois, self.n_channels, *self.output_size)
        self.assertEqual(d_output_shape, tuple(y_var.data.size()))

    def check_forward_functional(self, x_var, rois_var):
        """crosscheck forward with naive roi_pooling"""
        # Set scale to 1.0 given that casting may be different
        prev_scale = self.spatial_scale
        self.spatial_scale = 1.0
        y_var = roi_pooling_2d(
            x_var, rois_var, self.output_size,
            spatial_scale=self.spatial_scale)
        d_y = roi_pooling_2d_naive(
            x_var, rois_var, self.output_size, self.spatial_scale)
        np.testing.assert_almost_equal(y_var.data.cpu().numpy(),
                                       d_y.data.cpu().numpy())
        self.spatial_scale = prev_scale

    def check_backward(self, x_var, rois_var):
        # gradchek takes a tuple of tensor as input, check if your gradient
        # evaluated with these tensors are close enough to numerical
        # approximations and returns True if they all verify this condition.
        input = (x_var, rois_var)
        self.assertTrue(
            gradcheck(roi_pooling_2d, input, eps=1e-6, atol=2e-2,
                      raise_exception=True))

    def test_forward_gpu(self):
        x_var = Variable(self.x.cuda())
        rois_var = Variable(self.rois.cuda(), requires_grad=False)
        self.check_forward(x_var, rois_var)
        self.check_forward_functional(x_var, rois_var)

    def test_backward_gpu(self):
        x_var = Variable(self.x.cuda(), requires_grad=True)
        rois_var = Variable(self.rois.cuda(), requires_grad=False)
        self.check_backward(x_var, rois_var)


if __name__ == '__main__':
    unittest.main()

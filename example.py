import torch
import numpy as np
from roi_pooling.functions.roi_pooling import roi_pooling_2d

# Data parameters and fixed-random ROIs
batch_size = 3
n_channels = 4
input_size = (12, 8)
output_size = (5, 7)
spatial_scale = 0.6
rois = torch.FloatTensor([
    [0, 1, 1, 6, 6],
    [2, 6, 2, 7, 11],
    [1, 3, 1, 5, 10],
    [0, 3, 3, 3, 3]
])

# Generate random input tensor
x_np = np.arange(batch_size * n_channels *
                 input_size[0] * input_size[1],
                 dtype=np.float32)
x_np = x_np.reshape((batch_size, n_channels, *input_size))
np.random.shuffle(x_np)

# torchify and gpu transfer
x = torch.from_numpy(2 * x_np / x_np.size - 1)
x = x.cuda()
rois = rois.cuda()

# Actual ROIpoling operation
y = roi_pooling_2d(x, rois, output_size,
                   spatial_scale=spatial_scale)
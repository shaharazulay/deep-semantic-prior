import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from collections import OrderedDict
from functools import partial

from _utils import torch_to_np


class ConvAutoencoder(nn.Module):
    """
    Main model.
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.feature_maps = OrderedDict()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def collect_feature_maps(model):
    """
    Apply hook to collect feature maps from every conv layer
    """
    def hook(module, input, output, key):
        if isinstance(module, nn.Conv2d):
            feature_maps = torch_to_np(output)
            num_kernels = feature_maps.shape[0]
            feature_maps = np.split(feature_maps, num_kernels, axis=0)
            
            model.feature_maps[key] = feature_maps
    
    for idx, layer in enumerate(model.encoder):
        # hook is applied at inference time (only on the encoder part)
        layer.register_forward_hook(partial(hook, key='conv2d_{}'.format(idx + 1)))
        
"""DeepGaze MR inference architecture with a feature-only VGG19 backbone.

Adapted from mtangemann/deepgazemr at
7beab05d96a36eb7b343995a10d5e91d46e52ddf (deepgazemr/model.py).
The checkpoint keys and inference operations are preserved. The unused VGG
classifier is never constructed, and torchvision is not imported.

Copyright 2020 Matthias Tangemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F


class DeepGazeMR(nn.Module):
    def __init__(self, center_bias=None):
        super().__init__()
        self.features = _Backbone()
        self.readout = _Readout()
        self.finalizer = _Finalizer()
        self.center_bias = center_bias
        self._window_length = 16

    def forward(self, clip, center_bias=None):
        center_bias = self._adapt_center_bias(clip.shape[-2:], center_bias)
        center_bias = center_bias.to(clip.device)
        features = self.features(clip)
        priority_map = self.readout(features)
        return self.finalizer(priority_map, center_bias)

    def _adapt_center_bias(self, size, center_bias=None):
        center_bias = center_bias if center_bias is not None else self.center_bias
        if center_bias is None:
            raise ValueError("No center bias given")
        center_bias = center_bias.unsqueeze(0).unsqueeze(0)
        if center_bias.shape[-2:] != size:
            center_bias = F.interpolate(center_bias, size)
            center_bias /= center_bias.sum()
        return center_bias.log()


class _FeatureVGG(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG19 configuration E, with its last pooling layer replaced by the
        # Identity used upstream. Meta construction avoids unused random data.
        layers = []
        channels = 3
        with torch.device("meta"):
            for block in ((64, 64), (128, 128), (256,) * 4, (512,) * 4, (512,) * 4):
                for output_channels in block:
                    layers.extend((nn.Conv2d(channels, output_channels, 3, padding=1),
                                   nn.ReLU(inplace=True)))
                    channels = output_channels
                layers.append(nn.MaxPool2d(2, 2))
        layers[-1] = nn.Identity()
        self.features = nn.Sequential(*layers)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = _FeatureVGG()
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
        self._mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self._std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def forward(self, clip):
        clip = (clip - self._mean.to(clip.device)) / self._std.to(clip.device)
        return self.vgg.features(clip)


class _Readout(nn.Sequential):
    def __init__(self):
        layers = OrderedDict()
        channels = (512, 32, 32, 1)
        for index in range(1, 4):
            layers[f"conv{index}"] = nn.Conv2d(channels[index - 1], channels[index], 1)
            if index < 3:
                layers[f"norm{index}"] = nn.GroupNorm(1, channels[index], affine=True)
                layers[f"nlin{index}"] = nn.Softplus()
        super().__init__(layers)

    def forward(self, features):
        return super().forward(features.mean(dim=0, keepdim=True))


class _Finalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gauss_x = _GaussianFilter1d(1, 16.0, 97)
        self.gauss_y = _GaussianFilter1d(0, 16.0, 97)
        self.center_bias_weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, priority_map, center_bias):
        out = F.interpolate(priority_map, size=center_bias.shape[-2:], mode="nearest")
        out = self.gauss_x(out)
        out = self.gauss_y(out)
        out = out + self.center_bias_weight * center_bias
        out = out.squeeze()
        return out - out.logsumexp(dim=(0, 1))


class _GaussianFilter1d(nn.Module):
    def __init__(self, dim, sigma, kernel_size):
        super().__init__()
        self.sigma = nn.Parameter(torch.Tensor([sigma]), requires_grad=False)
        mean = (kernel_size - 1) / 2
        grid = torch.arange(float(kernel_size)) - mean
        kernel_shape = [1] * 4
        kernel_shape[2 + dim] = kernel_size
        self.register_buffer("grid", grid.view(kernel_shape))
        padding = [0] * 4
        padding[dim * 2 + 1] = math.ceil((kernel_size - 1) / 2)
        padding[dim * 2] = math.ceil((kernel_size - 1) / 2)
        self.padding = tuple(reversed(padding))

    def forward(self, x):
        x = F.pad(x, self.padding, "replicate", 0.0)
        kernel = torch.exp(-0.5 * (self.grid / self.sigma) ** 2)
        kernel = kernel / kernel.sum()
        return F.conv2d(x, kernel)

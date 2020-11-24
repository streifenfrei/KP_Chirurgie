import torch
import torch.nn as nn
import numpy as np


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DecoderBlock(nn.Module):
    """
    Decoder block according to https://arxiv.org/abs/1606.00373.
    These blocks are used in the CSL paper (https://arxiv.org/abs/1703.10701)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.indices = None

        self.unpooling = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = conv5x5(in_channels, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.proj = conv5x5(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        proj = x = self.unpooling(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        proj = self.proj(proj)
        x.add(proj)
        x = self.relu2(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=False, groups=1,
                 base_width=64, dilation=1, stride=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.norm_layer = norm_layer

        if self.downsample:
            self.identity_layer = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes * Bottleneck.expansion,
                          kernel_size=1,
                          stride=self.stride,
                          bias=False),
                self.norm_layer(self.planes * Bottleneck.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.identity_layer(x)

        out += identity
        out = self.relu(out)

        return out

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                conv3x3(in_channels, middle_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv3x3(in_channels, middle_channels),
                nn.ReLU(inplace=True),
                conv3x3(middle_channels, out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=False, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        if self.downsample:
            stride = 2
        else:
            stride = 1
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
            identity = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes * Bottleneck.expansion,
                          kernel_size=1,
                          stride=2,
                          bias=False),
                self.norm_layer(self.planes * Bottleneck.expansion)
            )(x)

        out += identity
        out = self.relu(out)

        return out

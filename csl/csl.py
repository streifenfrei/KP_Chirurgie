from enum import IntEnum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataLoader import OurDataLoader, image_transform


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, sampling=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               groups=groups,
                               bias=False,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sampling = sampling
        self.stride = stride

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

        if self.sampling is not None:
            identity = self.sampling(x)

        out += identity
        out = self.relu(out)

        return out


class CSLNet(nn.Module):
    resNetLayers = [
        [3, 4, 6, 3],  # 50
        [3, 4, 23, 3],  # 101
        [3, 8, 36, 3]  # 152
    ]

    class Encoder(IntEnum):
        resNet50 = 0
        resNet101 = 1
        resNet152 = 2

    class Sampling(IntEnum):
        none = 0
        up = 1
        down = 2

    def __init__(self,
                 encoder: Encoder = Encoder.resNet50):
        super(CSLNet, self).__init__()
        self.inplanes = 64
        self.base_width = 64
        self.dilation = 1
        self.groups = 1
        self.norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = self.resNetLayers[encoder]

        self.encoding_layer1 = self._make_layer(64, layers[0])
        self.encoding_layer2 = self._make_layer(128, layers[1])
        self.encoding_layer3 = self._make_layer(256, layers[2])
        self.encoding_layer4 = self._make_layer(512, layers[3])

    def _make_layer(self, planes, blocks, stride=2, sampling: Sampling=Sampling.none):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                self.norm_layer(planes * Bottleneck.expansion)
            )

        layers = [
            Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                             self.base_width, self.dilation, self.norm_layer)
        ]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                     base_width=self.base_width, dilation=self.dilation,
                                     norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.encoding_layer1(x)
        print(x.shape)
        x = self.encoding_layer2(x)
        print(x.shape)


        x = self.encoding_layer3(x)
        print(x.shape)

        x = self.encoding_layer4(x)
        print(x.shape)

        return x


if __name__ == '__main__':
    loader = DataLoader(
        dataset=OurDataLoader(data_dir=r'../dataset', transform=image_transform(p=1)),
        shuffle=True,
        batch_size=1,
        pin_memory=torch.cuda.is_available()
    )
    model = CSLNet()

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))

    for batchX, batchY in loader:
        out = model(batchX)
        #print(batchX.shape)
        #print(out.shape)

from enum import IntEnum

import torch
from torch.utils.data import DataLoader

from csl.net_modules import *
from dataLoader import OurDataLoader, image_transform


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

    class _Sampling(IntEnum):
        none = 0
        none_bottleneck = 1
        up = 2
        down = 3

    def __init__(self,
                 encoder: Encoder = Encoder.resNet50,
                 segmentation_classes=3,
                 localisation_classes=4):
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

        self.encoding_layer1 = self._make_layer(256, sampling=self._Sampling.none_bottleneck,  bottleneck_blocks=layers[0])
        self.encoding_layer2 = self._make_layer(512, sampling=self._Sampling.down, bottleneck_blocks=layers[1])
        self.encoding_layer3 = self._make_layer(1024, sampling=self._Sampling.down, bottleneck_blocks=layers[2])
        self.encoding_layer4 = self._make_layer(2048, sampling=self._Sampling.down,  bottleneck_blocks=layers[3])

        self.bottleneck_layer = self._make_layer(1024, sampling=self._Sampling.none)

        self.decoding_layer1_1 = self._make_layer(512, sampling=self._Sampling.up)
        self.inplanes = 1024
        self.decoding_layer1_2 = self._make_layer(512, sampling=self._Sampling.none)
        self.decoding_layer2_1 = self._make_layer(256, sampling=self._Sampling.up)
        self.inplanes = 512
        self.decoding_layer2_2 = self._make_layer(256, sampling=self._Sampling.none)
        self.decoding_layer3_1 = self._make_layer(128, sampling=self._Sampling.up)
        self.inplanes = 256
        self.decoding_layer3_2 = self._make_layer(128, sampling=self._Sampling.none)
        self.decoding_layer4 = self._make_layer(64, sampling=self._Sampling.up)

        self.segmentation_layer = self._make_layer(segmentation_classes, sampling=self._Sampling.up)
        self.inplanes = 64
        self.pre_localisation_layer = self._make_layer(32, sampling=self._Sampling.up)
        self.localisation_layer = self._make_layer(localisation_classes, sampling=self._Sampling.none)

    def _make_decoder_block(self, planes):
        return DecoderBlock(self.inplanes, self.inplanes, planes)

    def _make_bottleneck_block(self, planes, downsample=False, stride=1):
        return Bottleneck(self.inplanes, planes,
                          downsample=downsample,
                          groups=self.groups,
                          base_width=self.base_width,
                          dilation=self.dilation,
                          stride=stride,
                          norm_layer=self.norm_layer)

    def _make_layer(self, planes, sampling: _Sampling = _Sampling.none, bottleneck_blocks=2):
        layers = []
        if sampling == self._Sampling.down:
            bottleneck_planes = int(planes / Bottleneck.expansion)
            layers.append(self._make_bottleneck_block(bottleneck_planes, downsample=True, stride=2))
            self.inplanes = planes
            for _ in range(1, bottleneck_blocks):
                layers.append(self._make_bottleneck_block(bottleneck_planes))
        elif sampling == self._Sampling.up:
            layers.append(self._make_decoder_block(planes))
            self.inplanes = planes
        elif sampling == self._Sampling.none_bottleneck:
            bottleneck_planes = int(planes / Bottleneck.expansion)
            layers.append(self._make_bottleneck_block(bottleneck_planes, downsample=True, stride=1))
            self.inplanes = planes
            for _ in range(1, bottleneck_blocks):
                layers.append(self._make_bottleneck_block(bottleneck_planes))
        else:
            layers.append(conv3x3(self.inplanes, planes))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # encoder
        x = enc1 = self.encoding_layer1(x)
        x = enc2 = self.encoding_layer2(x)
        x = enc3 = self.encoding_layer3(x)
        x = self.encoding_layer4(x)

        # bottleneck
        x = self.bottleneck_layer(x)

        # decoder
        x = self.decoding_layer1_1(x)
        dec1_2 = self.decoding_layer1_2(enc3)
        x += dec1_2
        x = self.decoding_layer2_1(x)
        dec2_2 = self.decoding_layer2_2(enc2)
        x += dec2_2
        x = self.decoding_layer3_1(x)
        dec3_2 = self.decoding_layer3_2(enc1)
        x += dec3_2
        x = self.decoding_layer4(x)

        # csl part
        seg = self.segmentation_layer(x)
        x = self.pre_localisation_layer(x)
        x = torch.cat((seg, x))
        x = self.localisation_layer(x)
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

    for step, (batchX, batchY) in enumerate(loader):
        if step == 1:
            out = model(batchX)

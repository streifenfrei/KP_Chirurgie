from enum import IntEnum

from detectron2.layers import ROIAlign
from torch import nn
import torch

from csl.net_modules import conv3x3, DecoderBlock, conv1x1


class Decoder(nn.Module):
    class _Sampling(IntEnum):
        none_relu = 0
        none_norm = 1
        up = 2

    def __init__(self, localisation_classes, dropout=0.5):
        super().__init__()
        # the layers in our decoding head have only 256 input channels, due to the feature maps outputted by the FPN
        # (the original architecture halves the channel count in every upsampling step starting with 2048,
        # which is common in UNET architectures)
        self.bottleneck_layer = self._make_layer(256, 256, sampling=self._Sampling.none_norm)
        self.decoding_layer1_1 = self._make_layer(256, 256, sampling=self._Sampling.up)
        self.decoding_layer1_2 = self._make_layer(256, 256, sampling=self._Sampling.none_norm)
        self.dropout_de1 = nn.Dropout(p=dropout)

        self.decoding_layer2_1 = self._make_layer(256, 256, sampling=self._Sampling.up)
        self.decoding_layer2_2 = self._make_layer(256, 256, sampling=self._Sampling.none_norm)
        self.dropout_de2 = nn.Dropout(p=dropout)

        self.decoding_layer3_1 = self._make_layer(256, 128, sampling=self._Sampling.up)
        self.decoding_layer3_2 = self._make_layer(256, 128, sampling=self._Sampling.none_norm)
        self.dropout_de3 = nn.Dropout(p=dropout)

        self.decoding_layer4 = self._make_layer(128, 64, sampling=self._Sampling.up)
        self.dropout_de4 = nn.Dropout(p=dropout)

        self.segmentation_layer = conv3x3(64, 1)
        self.pre_localisation_layer = self._make_layer(64, 32, sampling=self._Sampling.none_relu)
        self.localisation_layer = self._make_layer(33, localisation_classes, sampling=self._Sampling.none_relu)

    def _make_layer(self, inplanes, outplanes, sampling: _Sampling = _Sampling.none_norm):
        block = None
        if sampling == self._Sampling.up:
            block = DecoderBlock(inplanes, outplanes)
        elif sampling == self._Sampling.none_norm:
            block = nn.Sequential(conv1x1(inplanes, outplanes), nn.BatchNorm2d(outplanes))
        elif sampling == self._Sampling.none_relu:
            block = nn.Sequential(conv3x3(inplanes, outplanes), nn.LeakyReLU(inplace=True))
        return block

    def forward(self, features):
        x, feature1, feature2, feature3 = features

        x = self.bottleneck_layer(x)

        x = self.decoding_layer1_1(x)
        dec1_2 = self.decoding_layer1_2(feature1)
        x = x.add(dec1_2)
        x = self.dropout_de1(x)

        x = self.decoding_layer2_1(x)
        dec2_2 = self.decoding_layer2_2(feature2)
        x = x.add(dec2_2)
        x = self.dropout_de2(x)

        x = self.decoding_layer3_1(x)
        dec3_2 = self.decoding_layer3_2(feature3)
        x = x.add(dec3_2)
        x = self.dropout_de3(x)

        x = self.decoding_layer4(x)
        x = self.dropout_de4(x)

        # csl part
        segmentation = self.segmentation_layer(x)
        x = self.pre_localisation_layer(x)
        x = torch.cat((segmentation, x), 1)
        localisation = self.localisation_layer(x)
        return segmentation, localisation

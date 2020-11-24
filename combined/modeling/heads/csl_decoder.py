from enum import IntEnum

import typing
from torch import nn
import torch

from csl.net_modules import conv3x3, DecoderBlock, conv1x1


class Decoder(nn.Module):
    def __init__(self, segmentation_classes: int, localisation_classes: int, device: str, dropout: float = 0.5):
        super().__init__()
        # one decoder per segmentation class
        self.decoder_heads = nn.ModuleList(
            [DecoderHead(localisation_classes, dropout=dropout).to(device) for i in range(segmentation_classes)])

    def forward(self, features: typing.List[typing.List[torch.Tensor]],
                classes_per_image: typing.List[typing.List[int]]):
        futures_lists: typing.List[typing.List[torch.jit.Future[typing.Tuple[torch.Tensor, torch.Tensor]]]] = []
        box = 0
        for classes in classes_per_image:
            futures: typing.List[torch.jit.Future[typing.Tuple[torch.Tensor, torch.Tensor]]] = []
            current_features = [i[box] for i in features]
            box += 1
            for cls in classes:
                for index, decoder_head in enumerate(self.decoder_heads):
                    if cls == index:
                        future = torch.jit.fork(decoder_head, current_features)  # call asynchronously
                        futures.append(future)
            futures_lists.append(futures)
        # get results of asynchronous tasks
        output: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]] = []
        for futures in futures_lists:
            segs = []
            locs = []
            for future in futures:
                seg, loc = torch.jit.wait(future)
                segs.append(seg)
                locs.append(loc)
            if len(segs) != 0 and len(locs) != 0:
                seg = torch.cat(segs)
                loc = torch.cat(locs)
                output.append((seg, loc))
        return output

class DecoderHead(nn.Module):
    class _Sampling(IntEnum):
        none_relu = 0
        none_norm = 1
        up = 2

    def __init__(self, localisation_classes: int, dropout: float = 0.5):
        super().__init__()
        # the layers in our decoding head have only 256 input channels, due to the feature maps output by the FPN
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
        self.localisation_layer = conv3x3(33, localisation_classes)

    def _make_layer(self, inplanes: int, outplanes: int, sampling: _Sampling = _Sampling.none_norm):
        block = None
        if sampling == self._Sampling.up:
            block = DecoderBlock(inplanes, outplanes)
        elif sampling == self._Sampling.none_norm:
            block = nn.Sequential(conv1x1(inplanes, outplanes), nn.BatchNorm2d(outplanes))
        elif sampling == self._Sampling.none_relu:
            block = nn.Sequential(conv3x3(inplanes, outplanes), nn.LeakyReLU(inplace=True))
        return block

    def forward(self, features: typing.List[torch.Tensor]):
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

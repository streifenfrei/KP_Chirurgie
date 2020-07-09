from enum import IntEnum

from detectron2.utils.registry import Registry
from torch import nn

from csl.net_modules import conv3x3, Bottleneck, DecoderBlock, conv1x1

CSL_HEAD_REGISTRY = Registry("CSL_HEAD")

@CSL_HEAD_REGISTRY.register()
class CSLHead(nn.Module):

    class _Sampling(IntEnum):
        none_relu = 0
        none_norm = 1
        up = 2

    def __init__(self, cfg, input_shape):
        super().__init__()
        localisation_classes = cfg.MODEL.CSL_HEAD.LOCALISATION_CLASSES
        self.bottleneck_layer = self._make_layer(2048, 1024, sampling=self._Sampling.none_norm)

        self.decoding_layer1_1 = self._make_layer(256, 512, sampling=self._Sampling.up)
        self.decoding_layer1_2 = self._make_layer(1024, 512, sampling=self._Sampling.none_norm)
        self.decoding_layer2_1 = self._make_layer(256, 256, sampling=self._Sampling.up)
        self.decoding_layer2_2 = self._make_layer(512, 256, sampling=self._Sampling.none_norm)
        self.decoding_layer3_1 = self._make_layer(256, 128, sampling=self._Sampling.up)
        self.decoding_layer3_2 = self._make_layer(256, 128, sampling=self._Sampling.none_norm)
        self.decoding_layer4 = self._make_layer(128, 64, sampling=self._Sampling.up)

        self.segmentation_layer = conv3x3(64, 1)
        self.pre_localisation_layer = self._make_layer(64, 32, sampling=self._Sampling.none_relu)
        self.localisation_layer = self._make_layer(33, localisation_classes, sampling=self._Sampling.none_relu)
        _dropout = 0.5

        self.dropout_de1 = nn.Dropout(p=_dropout)
        self.dropout_de2 = nn.Dropout(p=_dropout)
        self.dropout_de3 = nn.Dropout(p=_dropout)
        self.dropout_de4 = nn.Dropout(p=_dropout)

        return

    def _make_layer(self, inplanes, outplanes, sampling: _Sampling = _Sampling.none_norm):
        block = None
        if sampling == self._Sampling.up:
            block = DecoderBlock(inplanes, outplanes)
        elif sampling == self._Sampling.none_norm:
            block = nn.Sequential(conv1x1(inplanes, outplanes), nn.BatchNorm2d(outplanes))
        elif sampling == self._Sampling.none_relu:
            block = nn.Sequential(conv3x3(inplanes, outplanes), nn.ReLU(inplace=True))
        return block


def build_csl_head(cfg, input_shape):
    name = cfg.MODEL.CSL_HEAD.NAME
    return CSL_HEAD_REGISTRY.get(name)(cfg, input_shape)

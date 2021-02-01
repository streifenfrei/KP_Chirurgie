from csl.net import CSLNet
from csl.net_modules import *


class QuantizableDecoderBlock(DecoderBlock):
    def __init__(self, in_channels, out_channels, indices_shape):
        super().__init__(in_channels, out_channels)
        self.indices = construct_indices(torch.zeros(indices_shape))
        self.indices.requires_grad = False

    def forward(self, x):
        proj = x = self.unpooling(x, self.indices)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        proj = self.proj(proj)
        x.add(proj)
        x = self.relu2(x)
        return x


class QuantizableCSLNet(CSLNet):
    def __init__(self,
                 input_shape,
                 encoder: CSLNet.Encoder = CSLNet.Encoder.res_net_50,
                 localisation_classes=4):
        self._indices_shape = input_shape
        super().__init__(encoder, localisation_classes)
        self.inplanes = 1024
        self._indices_shape = (input_shape[0], 1024, int(input_shape[2] / 32), int(input_shape[3] / 32))
        self.decoding_layer1_1 = self._make_layer(512, sampling=self._Sampling.up)
        self._indices_shape = (input_shape[0], 512, int(input_shape[2] / 16), int(input_shape[3] / 16))
        self.decoding_layer2_1 = self._make_layer(256, sampling=self._Sampling.up)
        self._indices_shape = (input_shape[0], 256, int(input_shape[2] / 8), int(input_shape[3] / 8))
        self.decoding_layer3_1 = self._make_layer(128, sampling=self._Sampling.up)
        self._indices_shape = (input_shape[0], 128, int(input_shape[2] / 4), int(input_shape[3] / 4))
        self.decoding_layer4 = self._make_layer(64, sampling=self._Sampling.up)

    def _make_layer(self, planes, sampling: CSLNet._Sampling = CSLNet._Sampling.none_norm, bottleneck_blocks=2, update_planes=True):
        inplanes = self.inplanes
        block = super()._make_layer(planes, sampling, bottleneck_blocks, update_planes)
        if sampling == self._Sampling.up:
            block = QuantizableDecoderBlock(inplanes, planes, self._indices_shape)
        return block

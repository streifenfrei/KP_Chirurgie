from enum import IntEnum

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import distance

from csl.net_modules import *
from dataLoader import OurDataLoader, image_transform


class CSLNet(nn.Module):
    _res_net_layers = [
        [3, 4, 6, 3],  # 50
        [3, 4, 23, 3],  # 101
        [3, 8, 36, 3]  # 152
    ]

    class Encoder(IntEnum):
        res_net_50 = 0
        res_net_101 = 1
        res_net_152 = 2

    class _Sampling(IntEnum):
        none = 0
        none_bottleneck = 1
        up = 2
        down = 3

    def __init__(self,
                 encoder: Encoder = Encoder.res_net_50,
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

        layers = self._res_net_layers[encoder]

        self.encoding_layer1 = self._make_layer(256, sampling=self._Sampling.none_bottleneck,  bottleneck_blocks=layers[0])
        self.encoding_layer2 = self._make_layer(512, sampling=self._Sampling.down, bottleneck_blocks=layers[1])
        self.encoding_layer3 = self._make_layer(1024, sampling=self._Sampling.down, bottleneck_blocks=layers[2])
        self.encoding_layer4 = self._make_layer(2048, sampling=self._Sampling.down,  bottleneck_blocks=layers[3])

        self.bottleneck_layer = self._make_layer(1024, sampling=self._Sampling.none)

        self.decoding_layer1_1 = self._make_layer(512, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer1_2 = self._make_layer(512, sampling=self._Sampling.none)
        self.decoding_layer2_1 = self._make_layer(256, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer2_2 = self._make_layer(256, sampling=self._Sampling.none)
        self.decoding_layer3_1 = self._make_layer(128, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer3_2 = self._make_layer(128, sampling=self._Sampling.none)
        self.decoding_layer4 = self._make_layer(64, sampling=self._Sampling.up)

        self.segmentation_layer = self._make_layer(segmentation_classes, sampling=self._Sampling.up, update_planes=False)
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

    def _make_layer(self, planes, sampling: _Sampling = _Sampling.none, bottleneck_blocks=2, update_planes=True):
        layers = []
        old_inplanes = self.inplanes
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

        if not update_planes:
            self.inplanes = old_inplanes

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
        segmentation = self.segmentation_layer(x)
        x = self.pre_localisation_layer(x)
        x = torch.cat((segmentation, x))
        localisation = self.localisation_layer(x)
        return segmentation, localisation


class GaussianFunction:
    def __init__(self, sigma):
        self.sigma = sigma
        self.a = 1 / (sigma * np.sqrt(2 * np.pi))
        self.b = (2 * (self.sigma ** 2))

    def __call__(self, x):
        gaussian = np.exp(-x / self.b)
        return self.a * gaussian


def loss_function(output, target, gaussian_function: GaussianFunction, lambdah=1):
    output_segmentation, output_localisation = output
    target_segmentation, target_localisation = target
    # segmentation
    segmentation_loss_function = nn.CrossEntropyLoss()
    segmentation_loss = segmentation_loss_function(output_segmentation, target_segmentation)
    # localization
    batch_size, localisation_classes, height, width = list(output_localisation.shape)
    target_localisation_array = np.zeros([batch_size, localisation_classes, height, width])
    for batch, localisation_class, y, x in np.nditer(target_localisation_array, flags=['multi_index']):
        # TODO retrieve target points
        target_points = [(2, 3), (31, 44)]
        target_value = 0
        if target_points:
            distances = []
            for target_point in target_points:
                euclidean_distance = distance.euclidean((x, y), target_point)
                distances.append(euclidean_distance)
            target_value = gaussian_function(min(distances))
        target_localisation_array[batch, localisation_class, y, x] = target_value
    target_localisation_tensor = torch.tensor(target_localisation_array)
    localisation_loss_function = nn.MSELoss()
    localisation_loss = localisation_loss_function(output_localisation, target_localisation_tensor)
    return segmentation_loss + (lambdah * localisation_loss)


def train(model: CSLNet, train_loader, sigma=5, lambdah=1, epochs=20, learning_rate=0.01, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gaussian_function = GaussianFunction(sigma)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = loss_function(output, target, gaussian_function, lambdah)
            loss.backward()
            optimizer.step()
        # TODO validation
    model.eval()


if __name__ == '__main__':
    loader = DataLoader(
        dataset=OurDataLoader(data_dir=r'../dataset', transform=image_transform(p=1)),
        shuffle=True,
        batch_size=1,
        pin_memory=torch.cuda.is_available()
    )

    model = CSLNet()
    for step, (batchX, batchY) in enumerate(loader):
        if step == 1:
            print(batchX.shape)
            out = model(batchX)

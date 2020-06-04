import os
from enum import IntEnum

from csl.net_modules import *
from dataLoader import train_val_dataset


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
                 segmentation_classes=4,
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

        self.encoding_layer1 = self._make_layer(256, sampling=self._Sampling.none_bottleneck,
                                                bottleneck_blocks=layers[0])
        self.encoding_layer2 = self._make_layer(512, sampling=self._Sampling.down, bottleneck_blocks=layers[1])
        self.encoding_layer3 = self._make_layer(1024, sampling=self._Sampling.down, bottleneck_blocks=layers[2])
        self.encoding_layer4 = self._make_layer(2048, sampling=self._Sampling.down, bottleneck_blocks=layers[3])

        self.bottleneck_layer = self._make_layer(1024, sampling=self._Sampling.none)

        self.decoding_layer1_1 = self._make_layer(512, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer1_2 = self._make_layer(512, sampling=self._Sampling.none)
        self.decoding_layer2_1 = self._make_layer(256, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer2_2 = self._make_layer(256, sampling=self._Sampling.none)
        self.decoding_layer3_1 = self._make_layer(128, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer3_2 = self._make_layer(128, sampling=self._Sampling.none)
        self.decoding_layer4 = self._make_layer(64, sampling=self._Sampling.up)

        self.segmentation_layer = self._make_layer(segmentation_classes, sampling=self._Sampling.none,
                                                   update_planes=False)
        self.pre_localisation_layer = self._make_layer(32, sampling=self._Sampling.none)
        self.inplanes = 32 + segmentation_classes
        self.localisation_layer = self._make_layer(localisation_classes, sampling=self._Sampling.none)

        self.segmentation_classes = segmentation_classes
        self.localisation_classes = localisation_classes

    def _make_decoder_block(self, planes):
        return DecoderBlock(self.inplanes, planes)

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
            layers.append(nn.Sequential(conv1x1(self.inplanes, planes), nn.BatchNorm2d(planes)))
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
        x.add(dec1_2)
        x = self.decoding_layer2_1(x)
        dec2_2 = self.decoding_layer2_2(enc2)
        x.add(dec2_2)
        x = self.decoding_layer3_1(x)
        dec3_2 = self.decoding_layer3_2(enc1)
        x.add(dec3_2)
        x = self.decoding_layer4(x)

        # csl part
        segmentation = self.segmentation_layer(x)
        x = self.pre_localisation_layer(x)
        x = torch.cat((segmentation, x), 1)
        localisation = self.localisation_layer(x)
        return segmentation, localisation


def loss_function(output, target, lambdah=1):
    output_segmentation, output_localisation = output
    target_segmentation, target_localisation = target
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    output_segmentation = upsample(output_segmentation)
    output_localisation = upsample(output_localisation)
    batch_size, localisation_classes, height, width = output_localisation.shape
    # segmentation
    segmentation_loss_function = nn.CrossEntropyLoss(reduction='mean')
    segmentation_loss = segmentation_loss_function(output_segmentation, target_segmentation)
    # localization
    localisation_loss_function = nn.MSELoss(reduction='sum')
    localisation_loss = localisation_loss_function(output_localisation, target_localisation) / (
            localisation_classes * batch_size)
    return segmentation_loss + (lambdah * localisation_loss), segmentation_loss.item(), localisation_loss.item()


def prepare_batch(batch, segmentation_classes, localisation_classes):
    inputs, target = batch
    target = target.permute(0, 3, 1, 2)
    target_segmentation, target_localisation = torch.split(target, [segmentation_classes, localisation_classes], dim=1)
    target_segmentation_np = np.array([np.argmax(a, axis=0) for a in target_segmentation.numpy()])
    target_segmentation = torch.tensor(target_segmentation_np)
    return inputs, (target_segmentation, target_localisation)


def prepare_datasets(datasets, segmentation_classes, localisation_classes):
    train_loader, val_loader = datasets
    new_train_loader = []
    new_val_loader = []
    for batch in train_loader:
        new_train_loader.append(prepare_batch(batch, segmentation_classes, localisation_classes))
    for batch in val_loader:
        new_val_loader.append(prepare_batch(batch, segmentation_classes, localisation_classes))
    return new_train_loader, new_val_loader


def _train_step(epoch, index, batch, model, lambdah, device, optimizer):
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    output = model(inputs)
    targets = (targets[0].to(device), targets[1].to(device))
    loss, segmentation_loss, localisation_loss = loss_function(output, targets, lambdah)
    loss.backward()
    optimizer.step()
    print(
        "training: epoch: {0} | batch: {1} | loss: {2} ({3} + {4} * {5})".format(epoch, index, loss, segmentation_loss,
                                                                                 lambdah, localisation_loss))


def _val_step(epoch, index, batch, model, lambdah, device):
    inputs, targets = batch
    inputs = inputs.to(device)
    output = model(inputs)
    targets = (targets[0].to(device), targets[1].to(device))
    loss, segmentation_loss, localisation_loss = loss_function(output, targets, lambdah)
    print("validation: epoch: {0} | batch: {1} | loss: {2} ({3} + {4} * {5})".format(epoch, index, loss,
                                                                                     segmentation_loss, lambdah,
                                                                                     localisation_loss))
    return ",{0}".format(str(loss.item()))


def train(model: CSLNet, dataset, optimizer, lambdah=1, start_epoch=0, max_epochs=1000000, save_rate=10,
          workspace='', device="cpu"):
    datasets = train_val_dataset(dataset, validation_split=0.3, train_batch_size=1,
                                 valid_batch_size=1, shuffle_dataset=True)
    train_loader, val_loader = prepare_datasets(datasets, model.segmentation_classes, model.localisation_classes)
    save_file = os.path.join(workspace, 'csl.pth')
    validation_file = os.path.join(workspace, 'csl_val.csv')
    validation_string = ''
    for epoch in range(start_epoch, max_epochs):
        # training
        model.train()
        for index, batch in enumerate(train_loader):
            _train_step(epoch, index, batch, model, lambdah, device, optimizer)
        # validation
        model.eval()
        validation_string += "\n{0}".format(str(epoch))
        for index, batch in enumerate(val_loader):
            validation_string += _val_step(epoch, index, batch, model, lambdah, device)
        # saving
        if not epoch % save_rate:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, save_file)
            with open(validation_file, 'a') as file:
                file.write(validation_string)
                validation_string = ''
            print("saved model.")
        print("\n")


def visualize(model: CSLNet, dataset, device='cpu'):
    loader = train_val_dataset(dataset, validation_split=0, train_batch_size=1,
                                valid_batch_size=1, shuffle_dataset=True)[0]

    model.eval()
    import matplotlib.pyplot as plt
    for batch in loader:
        fig = plt.figure(figsize=(12, 6))
        inputs, _ = batch
        fig.add_subplot(2, 5, 1)
        plt.imshow(inputs[0].view(inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]).permute(1, 2, 0))
        fig_counter = 2
        inputs = inputs.to(device)
        segmentation, localisation = model(inputs)
        segmentation = segmentation.cpu().detach()
        localisation = localisation.cpu().detach()

        batch_size, classes, width, height = list(segmentation.shape)
        for seg_class in range(classes):
            fig.add_subplot(2, 5, fig_counter)
            fig_counter += 1
            plt.imshow(segmentation[0, seg_class, :, :].view(width, height))

        batch_size, classes, width, height = list(localisation.shape)
        for loc_class in range(classes):
            fig.add_subplot(2, 5, fig_counter)
            fig_counter += 1
            plt.imshow(localisation[0, loc_class, :, :].view(width, height))
        plt.show()


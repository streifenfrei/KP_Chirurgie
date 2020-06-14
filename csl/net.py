import os
from enum import IntEnum
from torch.utils.tensorboard import SummaryWriter

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
        none_relu = 0
        none_norm = 1
        none_bottleneck = 2
        up = 3
        down = 4

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

        self.bottleneck_layer = self._make_layer(1024, sampling=self._Sampling.none_norm)

        self.decoding_layer1_1 = self._make_layer(512, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer1_2 = self._make_layer(512, sampling=self._Sampling.none_norm)
        self.decoding_layer2_1 = self._make_layer(256, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer2_2 = self._make_layer(256, sampling=self._Sampling.none_norm)
        self.decoding_layer3_1 = self._make_layer(128, sampling=self._Sampling.up, update_planes=False)
        self.decoding_layer3_2 = self._make_layer(128, sampling=self._Sampling.none_norm)
        self.decoding_layer4 = self._make_layer(64, sampling=self._Sampling.up)

        self.segmentation_layer = self._make_layer(segmentation_classes, sampling=self._Sampling.none_relu,
                                                   update_planes=False)
        self.pre_localisation_layer = self._make_layer(32, sampling=self._Sampling.none_relu)
        self.inplanes = 32 + segmentation_classes
        self.localisation_layer = self._make_layer(localisation_classes, sampling=self._Sampling.none_relu)

        self.segmentation_classes = segmentation_classes
        self.localisation_classes = localisation_classes

    def _make_bottleneck_block(self, planes, downsample=False, stride=1):
        return Bottleneck(self.inplanes, planes,
                          downsample=downsample,
                          groups=self.groups,
                          base_width=self.base_width,
                          dilation=self.dilation,
                          stride=stride,
                          norm_layer=self.norm_layer)

    def _make_layer(self, planes, sampling: _Sampling = _Sampling.none_norm, bottleneck_blocks=2, update_planes=True):
        block = None
        old_inplanes = self.inplanes
        if sampling == self._Sampling.down:
            layers = []
            bottleneck_planes = int(planes / Bottleneck.expansion)
            layers.append(self._make_bottleneck_block(bottleneck_planes, downsample=True, stride=2))
            self.inplanes = planes
            for _ in range(1, bottleneck_blocks):
                layers.append(self._make_bottleneck_block(bottleneck_planes))
            block = nn.Sequential(*layers)
        elif sampling == self._Sampling.up:
            block = DecoderBlock(self.inplanes, planes)
            self.inplanes = planes
        elif sampling == self._Sampling.none_bottleneck:
            layers = []
            bottleneck_planes = int(planes / Bottleneck.expansion)
            layers.append(self._make_bottleneck_block(bottleneck_planes, downsample=True, stride=1))
            self.inplanes = planes
            for _ in range(1, bottleneck_blocks):
                layers.append(self._make_bottleneck_block(bottleneck_planes))
            block = nn.Sequential(*layers)
        elif sampling == self._Sampling.none_norm:
            block = nn.Sequential(conv1x1(self.inplanes, planes), nn.BatchNorm2d(planes))
            self.inplanes = planes
        elif sampling == self._Sampling.none_relu:
            block = nn.Sequential(conv3x3(self.inplanes, planes), nn.ReLU(inplace=True))

        if not update_planes:
            self.inplanes = old_inplanes

        return block

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
        x = x.add(dec1_2)
        x = self.decoding_layer2_1(x)
        dec2_2 = self.decoding_layer2_2(enc2)
        x = x.add(dec2_2)
        x = self.decoding_layer3_1(x)
        dec3_2 = self.decoding_layer3_2(enc1)
        x = x.add(dec3_2)
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


def _reshape_filter_mask(mask: torch.Tensor):
    n, c, w, h = list(mask.shape)
    return nn.functional.pad(mask.reshape(n * c, 1, w, h), [1, 1, 1, 1], value=1)


def visualize_conv_filter(model: CSLNet, writer: SummaryWriter, global_step):
    # initial
    mask = _reshape_filter_mask(model.conv1.weight)
    writer.add_images("Encoder/Initial", mask, global_step=global_step)
    # enc1
    count = 1
    for child in model.encoding_layer1.children():
        if isinstance(child, Bottleneck):
            mask = _reshape_filter_mask(child.conv2.weight)
            writer.add_images("Encoder/1/{0}".format(str(count)), mask, global_step=global_step)
            count += 1
    # enc2
    count = 1
    for child in model.encoding_layer2.children():
        if isinstance(child, Bottleneck):
            mask = _reshape_filter_mask(child.conv2.weight)
            writer.add_images("Encoder/2/{0}".format(str(count)), mask, global_step=global_step)
            count += 1
    # enc3
    count = 1
    for child in model.encoding_layer3.children():
        if isinstance(child, Bottleneck):
            mask = _reshape_filter_mask(child.conv2.weight)
            writer.add_images("Encoder/3/{0}".format(str(count)), mask, global_step=global_step)
            count += 1
    # enc4
    count = 1
    for child in model.encoding_layer4.children():
        if isinstance(child, Bottleneck):
            mask = _reshape_filter_mask(child.conv2.weight)
            writer.add_images("Encoder/4/{0}".format(str(count)), mask, global_step=global_step)
            count += 1
    # dec1
    mask = _reshape_filter_mask(model.decoding_layer1_1.conv1.weight)
    writer.add_images("Decoder/1/1", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer1_1.conv2.weight)
    writer.add_images("Decoder/1/2", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer1_1.proj.weight)
    writer.add_images("Decoder/1/projection", mask, global_step=global_step)
    # dec2
    mask = _reshape_filter_mask(model.decoding_layer2_1.conv1.weight)
    writer.add_images("Decoder/2/1", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer2_1.conv2.weight)
    writer.add_images("Decoder/2/2", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer2_1.proj.weight)
    writer.add_images("Decoder/2/projection", mask, global_step=global_step)
    # dec3
    mask = _reshape_filter_mask(model.decoding_layer3_1.conv1.weight)
    writer.add_images("Decoder/3/1", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer3_1.conv2.weight)
    writer.add_images("Decoder/3/2", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer3_1.proj.weight)
    writer.add_images("Decoder/3/projection", mask, global_step=global_step)
    # dec1
    mask = _reshape_filter_mask(model.decoding_layer4.conv1.weight)
    writer.add_images("Decoder/4/1", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer4.conv2.weight)
    writer.add_images("Decoder/4/2", mask, global_step=global_step)
    mask = _reshape_filter_mask(model.decoding_layer4.proj.weight)
    writer.add_images("Decoder/4/projection", mask, global_step=global_step)
    # final
    mask = _reshape_filter_mask(next(model.segmentation_layer.children()).weight)
    writer.add_images("Out/segmentation", mask, global_step=global_step)
    mask = _reshape_filter_mask(next(model.pre_localisation_layer.children()).weight)
    writer.add_images("Out/pre localisation", mask, global_step=global_step)
    mask = _reshape_filter_mask(next(model.localisation_layer.children()).weight)
    writer.add_images("Out/localisation", mask, global_step=global_step)


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
    return loss.item()


def _val_step(epoch, index, batch, model, lambdah, device):
    inputs, targets = batch
    inputs = inputs.to(device)
    output = model(inputs)
    targets = (targets[0].to(device), targets[1].to(device))
    loss, segmentation_loss, localisation_loss = loss_function(output, targets, lambdah)
    print("validation: epoch: {0} | batch: {1} | loss: {2} ({3} + {4} * {5})".format(epoch, index, loss,
                                                                                     segmentation_loss, lambdah,
                                                                                     localisation_loss))
    return loss.item()


def train(model: CSLNet, dataset, optimizer, lambdah=1, start_epoch=0, max_epochs=1000000, save_rate=10,
          workspace='', device="cpu", batch_size=2):
    writer = SummaryWriter(log_dir=os.path.join(workspace, 'tensorboard'))
    datasets = train_val_dataset(dataset, validation_split=0.3, train_batch_size=batch_size,
                                 valid_batch_size=batch_size, shuffle_dataset=True)
    train_loader, val_loader = prepare_datasets(datasets, model.segmentation_classes, model.localisation_classes)
    save_file = os.path.join(workspace, 'csl.pth')
    # tensorboard
    for epoch in range(start_epoch, max_epochs):
        # training
        model.train()
        losses = []
        for index, batch in enumerate(train_loader):
            loss = _train_step(epoch, index, batch, model, lambdah, device, optimizer)
            losses.append(loss)
        writer.add_scalar('Loss/training', sum(losses) / len(losses), epoch)
        # validation
        model.eval()
        losses = []
        for index, batch in enumerate(val_loader):
            loss = _val_step(epoch, index, batch, model, lambdah, device)
            losses.append(loss)
            if epoch == start_epoch and index == 0:
                writer.add_graph(model, batch[0].to(device))
        writer.add_scalar('Loss/validation', sum(losses) / len(losses), epoch)
        # saving
        if not epoch % save_rate:
            # move old model to older_models directory
            if os.path.exists(save_file):
                model_directory = os.path.join(workspace, "older_models")
                if not os.path.exists(model_directory):
                    os.mkdir(model_directory)
                old_epoch = torch.load(save_file)['epoch']
                os.replace(save_file, os.path.join(model_directory, 'csl_{0}.pth'.format(old_epoch)))
            # save current model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, save_file)
            writer.flush()
            visualize_conv_filter(model, writer, epoch)
            print("saved model.")
        print("\n")


def visualize(model: CSLNet, dataset, device='cpu', batch_size=2):
    loader = train_val_dataset(dataset, validation_split=0, train_batch_size=batch_size,
                               valid_batch_size=batch_size, shuffle_dataset=True)[0]

    model.eval()
    import matplotlib.pyplot as plt
    for batch in loader:
        fig = plt.figure(figsize=(12, 6))
        inputs, target = batch
        fig.add_subplot(2, 5, 1)
        plt.imshow(inputs[0].view(inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]).permute(1, 2, 0))
        fig_counter = 2
        inputs = inputs.to(device)
        segmentation, localisation = model(inputs)
        segmentation = segmentation.cpu().detach()
        localisation = localisation.cpu().detach()

        batch_size, seg_classes, width, height = list(segmentation.shape)
        for seg_class in range(seg_classes):
            fig.add_subplot(2, 5, fig_counter)
            fig_counter += 1
            plt.imshow(segmentation[0, seg_class, :, :].view(width, height))

        batch_size, loc_classes, width, height = list(localisation.shape)
        for loc_class in range(loc_classes):
            fig.add_subplot(2, 5, fig_counter)
            fig_counter += 1
            plt.imshow(localisation[0, loc_class, :, :].view(width, height))
        plt.show()

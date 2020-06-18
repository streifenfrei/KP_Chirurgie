import os
import traceback
from enum import IntEnum
from torch.utils.tensorboard import SummaryWriter

from csl.net_modules import *
from dataLoader import train_val_dataset, OurDataLoader


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

        self.segmentation_layer = self._make_layer(1, sampling=self._Sampling.none_relu,
                                                   update_planes=False)
        self.pre_localisation_layer = self._make_layer(32, sampling=self._Sampling.none_relu)
        self.inplanes = 33
        self.localisation_layer = self._make_layer(localisation_classes, sampling=self._Sampling.none_relu)

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

    @staticmethod
    def _reshape_filter_mask(mask: torch.Tensor):
        n, c, w, h = list(mask.shape)
        return nn.functional.pad(mask.reshape(n * c, 1, w, h), [1, 1, 1, 1], value=1)

    def visualize(self, dataset, device='cpu', batch_size=2):
        loader = train_val_dataset(dataset, validation_split=0, train_batch_size=batch_size,
                                   valid_batch_size=batch_size, shuffle_dataset=True)[0]

        self.eval()
        import matplotlib.pyplot as plt
        for batch in loader:
            fig = plt.figure(figsize=(12, 6))
            inputs, target = batch
            fig.add_subplot(2, 5, 1)
            plt.imshow(inputs[0].view(inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]).permute(1, 2, 0))
            fig_counter = 2
            inputs = inputs.to(device)
            segmentation, localisation = self(inputs)
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


class Training:

    def __init__(self, model: CSLNet, dataset: OurDataLoader, optimizer,
                 segmentation_loss, lambdah: int = 1,
                 start_epoch: int = 0, max_epochs: int = 100000, save_rate: int = 10,
                 workspace: str = '', device: str = "cpu", batch_size: int = 2, validation_split=0.3):
        self.model = model
        self.workspace = workspace
        self.device = device

        self.datasets = train_val_dataset(dataset, validation_split=validation_split, train_batch_size=batch_size,
                                          valid_batch_size=batch_size, shuffle_dataset=True)
        self.optimizer = optimizer
        self.segmentation_loss = segmentation_loss
        self.lambdah = lambdah

        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.save_rate = save_rate

    class LossFunction:

        class SegmentationLoss(IntEnum):
            cross_entropy = 0
            dice = 1

        class _DiceLoss:

            def __call__(self, output, target):
                probs = nn.functional.softmax(output, dim=1)

                num = probs * target  # b,c,h,w--p*g
                num = torch.sum(num, dim=3)  # b,c,h
                num = torch.sum(num, dim=2)

                den1 = probs * probs  # --p^2
                den1 = torch.sum(den1, dim=3)  # b,c,h
                den1 = torch.sum(den1, dim=2)

                den2 = target * target  # --g^2
                den2 = torch.sum(den2, dim=3)  # b,c,h
                den2 = torch.sum(den2, dim=2)  # b,c

                dice = 2 * ((num + 10e-7) / (den1 + den2 + 10e-7))

                dice_eso = dice[:, 0:-1]  # we ignore bg dice val, and take the fg

                dice_total = 1 - torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

                return dice_total

        def __init__(self, segmentation_loss: SegmentationLoss, lambdah=1):
            self.segmentation_loss = segmentation_loss
            self.lambdah = lambdah

        def __call__(self, output, target):
            output_segmentation, output_localisation = output
            target_segmentation, target_localisation = target
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            output_segmentation = upsample(output_segmentation)
            output_localisation = upsample(output_localisation)
            batch_size, localisation_classes, height, width = output_localisation.shape
            # segmentation
            if self.segmentation_loss == self.SegmentationLoss.cross_entropy:
                segmentation_loss_function = nn.BCELoss(reduction='mean')
            elif self.segmentation_loss == self.SegmentationLoss.dice:
                segmentation_loss_function = self._DiceLoss()
            else:
                raise ValueError
            segmentation_loss = segmentation_loss_function(output_segmentation, target_segmentation)
            # localization
            localisation_loss_function = nn.MSELoss(reduction='sum')
            localisation_loss = localisation_loss_function(output_localisation, target_localisation) / (
                    localisation_classes * batch_size)
            return segmentation_loss + (
                    self.lambdah * localisation_loss), segmentation_loss.item(), localisation_loss.item()

    def _prepare_batch(self, batch):
        inputs, target = batch
        target = target.permute(0, 3, 1, 2)
        target_segmentation, target_localisation = torch.split(target, [1, self.model.localisation_classes], dim=1)
        return inputs, (target_segmentation, target_localisation)

    def _get_loss(self, batch):
        inputs, targets = self._prepare_batch(batch)
        inputs = inputs.to(self.device)
        output = self.model(inputs)
        targets = (targets[0].to(self.device), targets[1].to(self.device))
        loss_function = self.LossFunction(self.segmentation_loss, self.lambdah)
        return loss_function(output, targets)

    def _train_step(self, epoch, index, batch):
        self.optimizer.zero_grad()
        loss, segmentation_loss, localisation_loss = self._get_loss(batch)
        loss.backward()
        self.optimizer.step()
        print(
            "training: epoch: {0} | batch: {1} | loss: {2} ({3} + {4} * {5})".format(epoch, index, loss,
                                                                                     segmentation_loss,
                                                                                     self.lambdah, localisation_loss))
        return loss.item()

    def _val_step(self, epoch, index, batch):
        loss, segmentation_loss, localisation_loss = self._get_loss(batch)
        print("validation: epoch: {0} | batch: {1} | loss: {2} ({3} + {4} * {5})".format(epoch, index, loss,
                                                                                         segmentation_loss, self.lambdah,
                                                                                         localisation_loss))
        return loss.item()

    def start(self):
        writer = SummaryWriter(log_dir=os.path.join(self.workspace, 'tensorboard'))
        try:
            train_loader, val_loader = self.datasets
            save_file = os.path.join(self.workspace, 'csl.pth')
            for epoch in range(self.start_epoch, self.max_epochs):
                # training
                self.model.train()
                losses = []
                for index, batch in enumerate(train_loader):
                    loss = self._train_step(epoch, index, batch)
                    losses.append(loss)
                writer.add_scalar('Loss/training', sum(losses) / len(losses), epoch)
                # validation
                self.model.eval()
                losses = []
                for index, batch in enumerate(val_loader):
                    loss = self._val_step(epoch, index, batch)
                    losses.append(loss)
                    if epoch == self.start_epoch and index == 0:
                        writer.add_graph(self.model, batch[0].to(self.device))
                writer.add_scalar('Loss/validation', sum(losses) / len(losses), epoch)
                # saving
                if not epoch % self.save_rate:
                    # move old model to older_models directory
                    if os.path.exists(save_file):
                        model_directory = os.path.join(self.workspace, "older_models")
                        if not os.path.exists(model_directory):
                            os.mkdir(model_directory)
                        old_epoch = torch.load(save_file)['epoch']
                        os.replace(save_file, os.path.join(model_directory, 'csl_{0}.pth'.format(old_epoch)))
                    # save current model
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch + 1,
                    }, save_file)
                    writer.flush()
                    print("saved model.")
                print("\n")
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()




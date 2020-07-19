import os
import traceback
from enum import IntEnum
from torch.utils.tensorboard import SummaryWriter

from csl.net_modules import *
from torch.optim.lr_scheduler import *
from dataLoader import train_val_dataset, OurDataLoader

# for evaluating the local result
from evaluate import findNN, plotOverlayImages
from skimage.transform import resize

class loc_model(nn.Module):
    def __init__(self, hidden_dim, pretrained=True):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3]) # 50

        self.hidden_dim = hidden_dim

        self.image_width = 512
        self.image_height = 960
        if pretrained:
            self.load_state_dict(models.resnet50(pretrained=True).state_dict())
        _dropout = 0.5 
        en_out_size = 2048 # needs testing
        
        # linear
        self.hidden1_fc = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2_fc = nn.Linear(self.hidden_dim // 2, 4*4) #each point: 2 binary class, 2 coordinates
        # dropout

        self.dropout_cnn0 = nn.Dropout(p=_dropout)
        self.dropout_cnn1 = nn.Dropout(p=_dropout)
        self.dropout_cnn2 = nn.Dropout(p=_dropout)

        self.dropout_fc = nn.Dropout(p=_dropout)


    '''
    # TODO: test            
    def show_all_result(self, dataset, device='cpu', batch_size=1):
        loader = train_val_dataset(dataset, validation_split=0, train_batch_size=batch_size,
                                   valid_batch_size=batch_size, shuffle_dataset=True)[0]
        self.eval()
        i = 0
        for batch in loader:
            i += 1
            inputs, target = batch

            inputs = inputs.to(device)
            segmentation, localisation = self(inputs)
            segmentation = segmentation.cpu().detach()
            localisation = localisation.cpu().detach()
            localisation = localisation.numpy()
            target = target.cpu().detach().numpy()

            batch_size, seg_classes, width, height = list(segmentation.shape)
            
            ori_img = inputs[0].view(inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]).permute(1, 2, 0).cpu().detach().numpy()
            #ori_img = resize(ori_img, (256, 480, 3))
            #print('ori_img.shape:', ori_img.shape)
            
            seg_image = (nn.Sigmoid()(segmentation[0, 0, :, :].view(width, height))).numpy()
            seg_image = resize(seg_image,(ori_img.shape[0], ori_img.shape[1]))
            print('seg_image.shape:', seg_image.shape)
            
            batch_size, loc_classes, width, height = list(localisation.shape)
            loc_images = []
            label_loc_images = []
            for loc_class_ in range(loc_classes):
                loc_image = localisation[0, loc_class_, :, :]
                label_loc_image = target[0, :, :, 1 + loc_class_]
                loc_image = resize(loc_image, (ori_img.shape[0], ori_img.shape[1]))
                loc_images.append(loc_image)
                label_loc_images.append(label_loc_image)
      
            plotOverlayImages(ori_img, seg_image, loc_images, label_loc_images, r'../out/' + str(i) + '.png')        

    def visualize(self, dataset, device='cpu', batch_size=2):
        loader = train_val_dataset(dataset, validation_split=0, train_batch_size=batch_size,
                                   valid_batch_size=batch_size, shuffle_dataset=True)[0]
        self.eval()
        import matplotlib.pyplot as plt
        i = 0
        for batch in loader:
            i += 1
            fig = plt.figure(figsize=(12, 9))
            inputs, target = batch
            fig.add_subplot(3, 4, 1)
            plt.imshow(inputs[0].view(inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]).permute(1, 2, 0))
            fig_counter = 2
            inputs = inputs.to(device)
            segmentation, localisation = self(inputs)
            segmentation = segmentation.cpu().detach()
            localisation = localisation.cpu().detach()
            localisation = localisation.numpy()

            batch_size, seg_classes, width, height = list(segmentation.shape)
            for seg_class in range(seg_classes):
                fig.add_subplot(3, 4, fig_counter)
                fig_counter += 1
                plt.imshow(nn.Sigmoid()(segmentation[0, seg_class, :, :].view(width, height)))

            batch_size, loc_classes, width, height = list(localisation.shape)
            print(batch_size, loc_classes, width, height)
            fig_counter = 5
            for loc_class_ in range(loc_classes):
                fig.add_subplot(3, 4, fig_counter)
                fig_counter += 1
                for row in localisation[0, loc_class_, :, :]:
                    #for col in row:
                    print(np.max(row))
                    print('\n')
                print('========')
                plt.imshow(localisation[0, loc_class_, :, :])
            print(target.shape)
            for loc_class_ in range(seg_classes, loc_classes + 1):
                fig.add_subplot(3, 4, fig_counter)
                fig_counter += 1
                plt.imshow(target[0, :, :, loc_class_])
            plt.savefig('test' + str(i) + '.png')
    '''
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # encoder
        x = enc1 = self.encoding_layer1(x)
        x = self.dropout_en1(x)
        x = enc2 = self.encoding_layer2(x)
        x = self.dropout_en2(x)
        x = enc3 = self.encoding_layer3(x)
        x = self.dropout_en3(x)

        x = self.encoding_layer4(x)
        x = self.dropout_en4(x)
        
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1) # batch, 2048
        
        # linear
        loc_out = self.hidden1_fc(torch.tanh(x))
        loc_out = self.dropout_fc(loc_out) # shall we add the dropout in fc?
        loc_out = self.hidden2_fc(loc_out)

        return loc_out


class Training:

    def __init__(self, model: CSLNet, dataset: OurDataLoader, optimizer, scheduler,
                 segmentation_loss, lambdah: int = 1,
                 start_epoch: int = 0, max_epochs: int = 100000, save_rate: int = 15,
                 workspace: str = '', device: str = "cpu", batch_size: int = 2, validation_split=0.1):
        self.model = model
        self.workspace = workspace
        self.device = device

        self.datasets = train_val_dataset(dataset, validation_split=validation_split, train_batch_size=batch_size,
                                          valid_batch_size=batch_size, shuffle_dataset=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.segmentation_loss = segmentation_loss
        self.lambdah = lambdah

        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.save_rate = save_rate

    class LossFunction:
        def __init__(self, lambdah=1):
            self.lambdah = lambdah

        def __call__(self,  out_loc, target):
            # 0,1 dim: cross_entropy
            # 2,3 din: mse
            batch_size, out_size = out_loc.shape
            print('loss function, out_loc.shape: ', out_loc.shape)
            # === split vector of 16 to 4 points ===
            target_p1, target_p2, target_p3, target_p4 = torch.split(target, [4,4,4,4], dim=1)
            out_loc_p1, out_loc_p2, out_loc_p3, out_loc_p4 = torch.split(out_loc, [4,4,4,4], dim=1)
            # === split vector of 16 to 4 points end ===

            # === split every point to 2 parts ===
            target_p1_bin, target_p1_coord = torch.split(target_p1, [2,2], dim=1)
            out_p1_bin, out_p1_coord = torch.split(out_loc_p1, [2,2], dim=1)

            target_p2_bin, target_p2_coord = torch.split(target_p2, [2,2], dim=1)
            out_p2_bin, out_p2_coord = torch.split(out_loc_p2, [2,2], dim=1)

            target_p3_bin, target_p3_coord = torch.split(target_p3, [2,2], dim=1)
            out_p3_bin, out_p3_coord = torch.split(out_loc_p3, [2,2], dim=1)

            target_p4_bin, target_p4_coord = torch.split(target_p4, [2,2], dim=1)
            out_p4_bin, out_p4_coord = torch.split(out_loc_p4, [2,2], dim=1)
            # === split every point to 2 parts end ===

            # === for 4 points binary loss ===
            loss_function_bin = nn.CrossEntropyLoss(reduction = 'mean')
            loss_p1_bin = loss_function_bin(out_p1_bin, target_p1_bin)
            loss_p2_bin = loss_function_bin(out_p2_bin, target_p2_bin)
            loss_p3_bin = loss_function_bin(out_p3_bin, target_p3_bin)
            loss_p4_bin = loss_function_bin(out_p4_bin, target_p4_bin)
            loss_binary = loss_p1_bin + loss_p2_bin + loss_p3_bin + loss_p4_bin
            # === 4 points binary loss end ===


            # === for 4 points coord loss ===
            loss_function_coord = nn.MSELoss(reduction='sum')

            mse_p1 = target_p1_bin[0] * loss_function_coord(out_p1_coord, target_p1_coord) 
            mse_p2 = target_p2_bin[0] * loss_function_coord(out_p2_coord, target_p2_coord)
            mse_p3 = target_p3_bin[0] * loss_function_coord(out_p3_coord, target_p3_coord)
            mse_p4 = target_p4_bin[0] * loss_function_coord(out_p4_coord, target_p4_coord)

            loss_coord = (mse_p1 + mse_p2 + mse_p3 + mse_p4)
            # === 4 points coord loss end ===

            loss_total = loss_binary + self.lambdah * loss_coord
            return loss_total, loss_binary, loss_coord

    def _prepare_batch(self, batch):
        inputs, target = batch
        return inputs, (target_segmentation, target_localisation)

    def _get_loss(self, batch):
        inputs, targets = self._prepare_batch(batch)
        inputs = inputs.to(self.device)
        output = self.model(inputs)
        targets = targets.to(self.device)
        loss_function = self.LossFunction(self.lambdah)
        return loss_function(output, targets)

    def _train_step(self, epoch, index, batch):
        self.optimizer.zero_grad()
        loss, binary_loss, coord_loss = self._get_loss(batch)
        loss.backward()
        self.optimizer.step()
        print(
            "training: epoch: {0} | batch: {1} | loss: {2} ({3} + {4} * {5})".format(epoch, index, loss,
                                                                                     binary_loss,
                                                                                     self.lambdah, coord_loss))
        return loss.item()

    def _val_step(self, epoch, index, batch):
        loss, binary_loss, coord_loss = self._get_loss(batch)
        print("validation: epoch: {0} | batch: {1} | loss: {2} ({3} + {4} * {5})".format(epoch, index, loss,
                                                                                         binary_loss,
                                                                                         self.lambdah,
                                                                                         coord_loss))
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
                validation_loss = sum(losses) / len(losses)
                writer.add_scalar('Loss/validation', validation_loss, epoch)
                self.scheduler.step(validation_loss)
                writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

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
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'epoch': epoch + 1
                    }, save_file)
                    writer.flush()
                    print("saved model.")
                print("\n")
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()

from enum import IntEnum

from detectron2.utils.registry import Registry
import matplotlib.pyplot as plt
from torch import nn
import torch

from csl.net_modules import conv3x3, DecoderBlock, conv1x1

CSL_HEAD_REGISTRY = Registry("CSL_HEAD")

@CSL_HEAD_REGISTRY.register()
class CSLHead(nn.Module):

    @staticmethod
    def _segmentation_loss(pred, target):
        pred = pred.squeeze()
        target = target.type(torch.DoubleTensor)
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

    @staticmethod
    def _localisation_loss(pred, target):
        weights = torch.where(target > 0.0001, torch.full_like(target, 5), torch.full_like(target, 1))
        all_mse = (pred - target)**2
        weighted_mse = all_mse * weights
        return weighted_mse.sum() / (target.size(0) * target.size(1))

    class _Sampling(IntEnum):
        none_relu = 0
        none_norm = 1
        up = 2

    def __init__(self, cfg, dropout=0.5):
        super().__init__()
        self.lambdaa = cfg.MODEL.CSL_HEAD.LAMBDA
        localisation_classes = cfg.MODEL.CSL_HEAD.LOCALISATION_CLASSES

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

    def forward(self, x, instances):

        x = self._forward_per_instance(x, instances)
        if self.training:
            seg = torch.cat([i[0] for i in x])
            loc = torch.cat([i[1] for i in x])
            gt_masks = []
            gt_locs = []
            for instances_per_image in instances:
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.proposal_boxes.tensor, seg.size(2)
                ).to(device=seg.device)
                gt_masks.append(gt_masks_per_image)
                #gt_locs_per_image = instances_per_image.gt_locs.crop_and_resize(
                #    instances_per_image.proposal_boxes.tensor, loc.size(2)
                #).to(device=loc.device)
                #gt_locs.append(gt_locs_per_image)
            gt_masks = torch.cat(gt_masks, dim=0)
            #gt_locs = torch.cat(gt_locs, dim=0)
            return {"loss_seg": CSLHead._segmentation_loss(seg, gt_masks)}
                    #"loss_loc": self.lambdaa * CSLHead._localisation_loss(seg, gt_locs)}
        else:
            for instances_per_image, (seg, loc) in zip(instances, x):
                instances_per_image.pred_masks = torch.sigmoid(seg)
                instances_per_image.pred_loc = loc
            return instances


    def _forward_per_instance(self, x, instances):
        output = []
        x = [i.split(1, 0) for i in x]
        box = 0
        for instance in instances:
            segs = []
            locs = []
            for box_per_inst in range(len(instance)):
                features = [i[box] for i in x]
                box += 1
                seg, loc = self._forward_single_roi(features)
                segs.append(seg)
                locs.append(loc)
            seg = torch.cat(segs)
            loc = torch.cat(locs)
            output.append((seg, loc))
        return output

    def _forward_single_roi(self, features):
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


def build_csl_head(cfg, dropout=0.5):
    name = cfg.MODEL.CSL_HEAD.NAME
    return CSL_HEAD_REGISTRY.get(name)(cfg, dropout=dropout)

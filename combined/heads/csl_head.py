from enum import IntEnum

from detectron2.layers import ROIAlign
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage
from fvcore.common.registry import Registry
from torch import nn
import torch
import numpy as np
from csl.net_modules import conv3x3, DecoderBlock, conv1x1
from dataLoader import max_gaussian_help
from evaluate import DiceCoefficient

CSL_HEAD_REGISTRY = Registry("CSL_HEAD")

@CSL_HEAD_REGISTRY.register()
class CSLHead(nn.Module):
    """
    Head module for generating aligned segmentation masks and localisation heatmaps, given aligned feature maps
    """
    def _segmentation_loss(self, pred, target_bool):
        pred = pred.squeeze()
        target = target_bool.double()
        # evaluation
        storage = get_event_storage()
        pred_bool = pred > 0.5
        storage.put_scalar("csl_segmentation/dice", DiceCoefficient(epsilon=self.epsilon)(pred_bool, target_bool))
        mask_incorrect = pred_bool != target_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = target_bool.sum().item()
        false_positive = (mask_incorrect & ~target_bool).sum().item() / max(
            target_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & target_bool).sum().item() / max(num_positive, 1.0)
        storage.put_scalar("csl_segmentation/accuracy", mask_accuracy)
        storage.put_scalar("csl_segmentation/false_positive", false_positive)
        storage.put_scalar("csl_segmentation/false_negative", false_negative)
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

    def _localisation_loss(self, pred, target):
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
        self.sigma = cfg.MODEL.CSL_HEAD.SIGMA
        self.epsilon = cfg.MODEL.CSL_HEAD.EVALUATION.EPSILON
        self.output_resolution = cfg.MODEL.CSL_HEAD.POOLER_RESOLUTION * 16
        localisation_classes = cfg.MODEL.CSL_HEAD.LOCALISATION_CLASSES

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

        self.csl_hm_align = ROIAlign((self.output_resolution, self.output_resolution),
                                     spatial_scale=1, sampling_ratio=0)

    def _make_layer(self, inplanes, outplanes, sampling: _Sampling = _Sampling.none_norm):
        block = None
        if sampling == self._Sampling.up:
            block = DecoderBlock(inplanes, outplanes)
        elif sampling == self._Sampling.none_norm:
            block = nn.Sequential(conv1x1(inplanes, outplanes), nn.BatchNorm2d(outplanes))
        elif sampling == self._Sampling.none_relu:
            block = nn.Sequential(conv3x3(inplanes, outplanes), nn.ReLU(inplace=True))
        return block

    def _preprocess_gt_heatmaps(self, instances):
        """
        generates heatmaps (with gaussian kernels applied) from given csl keypoint vectors, which are then aligned to
        the proposed boxes with a ROIAlign module
        Args:
            instances: Instances object containing the ground truth csl keypoints in .gt_keypoints and the ground truth
                proposal boxes in .proposal_boxes
        Returns:
            Tensor: (N, L, Hm, Hw) where N is the number of instances, L the number of localisation classes.
        """
        heatmaps = []
        for keypoints_per_instance, box in zip(instances.gt_keypoints.keypoints, instances.proposal_boxes):
            heatmaps_per_instance = []
            for keypoints_per_class in keypoints_per_instance:
                heatmap = np.zeros((instances.image_size[1], instances.image_size[0]))
                for x, y in keypoints_per_class:
                    heatmap[x, y] = 1
                heatmap = max_gaussian_help(heatmap, self.sigma, 1)
                heatmaps_per_instance.append(torch.from_numpy(heatmap.transpose((2, 1, 0))))
            # we align every heatmap tensor individually due to some weird bug resulting in a mismatch of
            # ground truth instances and its heatmaps, when doing the alignment at the end, on a
            # stacked tensor of all heatmaps
            box = convert_boxes_to_pooler_format([Boxes(box.unsqueeze(0))])
            heatmaps_per_instance = torch.cat(heatmaps_per_instance).unsqueeze(0).float()
            heatmaps.append(self.csl_hm_align(heatmaps_per_instance, box))
        return torch.cat(heatmaps)

    def _forward_per_instances(self, x, instances):
        output = []
        x = [i.split(1, 0) for i in x]
        box = 0
        for instances_per_image in instances:
            segs = []
            locs = []
            for _ in range(len(instances_per_image)):
                features = [i[box] for i in x]
                box += 1
                seg, loc = self._forward_single_instance(features)
                segs.append(seg)
                locs.append(loc)
            seg = torch.cat(segs)
            loc = torch.cat(locs)
            output.append((seg, loc))
        return output

    def _forward_single_instance(self, features):
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

    def forward(self, x, instances):
        x = self._forward_per_instances(x, instances)
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
                gt_locs.append(self._preprocess_gt_heatmaps(instances_per_image))
            gt_masks = torch.cat(gt_masks, dim=0)
            gt_locs = torch.cat(gt_locs, dim=0)
            return {"loss_seg": self._segmentation_loss(seg, gt_masks),
                    "loss_loc": self.lambdaa * self._localisation_loss(loc, gt_locs)}
        else:
            for instances_per_image, (seg, loc) in zip(instances, x):
                instances_per_image.pred_masks = torch.sigmoid(seg)
                instances_per_image.pred_loc = loc
            return instances


def build_csl_head(cfg, dropout=0.5):
    name = cfg.MODEL.CSL_HEAD.NAME
    return CSL_HEAD_REGISTRY.get(name)(cfg, dropout=dropout)

from detectron2.layers import ROIAlign
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage
from fvcore.common.registry import Registry
from torch import nn
import torch
import numpy as np

from combined.modeling.heads.csl_decoder import Decoder
from dataLoader import max_gaussian_help
from evaluate import DiceCoefficient, get_threshold_score
CSL_HEAD_REGISTRY = Registry("CSL_HEAD")

@CSL_HEAD_REGISTRY.register()
class CSLHead(nn.Module):
    """
    Head module for generating aligned segmentation masks and localisation heatmaps, given aligned feature maps
    """
    def _segmentation_loss(self, pred, target_bool):
        pred = pred.squeeze()
        target = target_bool.double()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
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
        storage.put_scalar("csl_segmentation/loss", loss.item())
        return loss

    def _localisation_loss(self, pred, target):

        target = target.to(pred.device)
        weights = torch.where(target > 0.1, torch.full_like(target, self.loc_weight), torch.full_like(target, 1))
        all_mse = (pred - target)**2
        weighted_mse = all_mse * weights
        loss = weighted_mse.sum() / (target.shape[0] * target.shape[1])
        # fancy debugging visualisation
        #import matplotlib
        #matplotlib.use("TkAgg")
        #import matplotlib.pyplot as plt
        #for mask, locs, weigh, all, wmse, mask2 in zip(pred.split(1,0), target.split(1,0), weights.split(1,0), all_mse.split(1,0), weighted_mse.split(1,0)):
        #    fig = plt.figure(figsize=(24, 12))
        #    index = 1
        #    for loc1, loc2, loc3, loc4, loc5 in zip(locs.split(1, 1), mask.split(1,1), weigh.split(1,1), all.split(1,1), wmse.split(1,1)):
        #        fig.add_subplot(6,4,index+4)
        #        print(torch.max(loc1).item())
        #        plt.imshow(loc1.squeeze().detach())
        #        fig.add_subplot(6, 4, index+8)
        #        print(torch.max(loc2).item())
        #        plt.imshow(loc2.squeeze().detach())
        #        fig.add_subplot(6, 4, index + 12)
        #        print(torch.max(loc3).item())
        #        plt.imshow(loc3.squeeze().detach())
        #        fig.add_subplot(6, 4, index + 16)
        #        print(torch.max(loc4).item())
        #        plt.imshow(loc4.squeeze().detach())
        #        fig.add_subplot(6, 4, index + 20)
        #        print(torch.max(loc5).item())
        #        print("\n")
        #        plt.imshow(loc5.squeeze().detach())
        #        index += 1
        #    plt.show()

        #evaluation
        storage = get_event_storage()
        image_pairs = []
        for pred_single, target_single in zip(pred.split(1, 0), target.split(1, 0)):
            for pred_class, target_class in zip(pred_single.split(1, 1), target_single.split(1, 1)):
                pred_np = pred_class.squeeze().detach().cpu().numpy()
                target_np = target_class.squeeze().detach().cpu().numpy()
                image_pairs.append((target_np, pred_np))
        threshold_score, true_positive, false_positive, false_negative \
            = get_threshold_score(image_pairs, self.threshold_list)
        """
        if sum(threshold_score) > 0:
            score = []
            for i, score_count in enumerate(threshold_score):
                for j in range(score_count):
                    score.append(self.threshold_list[i])
            storage.put_histogram("csl_localisation/treshold_score", torch.tensor(score), bins=len(threshold_score))
        """ 
        epsilon = 10e-6
        precision = float(true_positive) / (true_positive + false_positive + epsilon)
        recall = float(true_positive) / (true_positive * false_negative + epsilon)
        f1 = 2 / ((1/(recall + epsilon)) + (1/(precision + epsilon)))
        storage.put_scalar("csl_localisation/precision", precision)
        storage.put_scalar("csl_localisation/recall", recall)
        storage.put_scalar("csl_localisation/f1", f1)
        storage.put_scalar("csl_localisation/loss", loss.item())
        return loss

    def __init__(self, cfg):
        super().__init__()
        self.hm_preprocessing_type = cfg.MODEL.CSL_HEAD.HM_PREPROCESSING
        self.lambdaa = cfg.MODEL.CSL_HEAD.LAMBDA
        self.sigma = cfg.MODEL.CSL_HEAD.SIGMA
        self.epsilon = cfg.MODEL.CSL_HEAD.EVALUATION.EPSILON
        self.loc_weight = cfg.MODEL.CSL_HEAD.LOC_WEIGHT
        self.threshold_list = cfg.MODEL.CSL_HEAD.EVALUATION.THRESHOLD_SCORE_LIST
        self.device = cfg.MODEL.DEVICE
        self.output_resolution = cfg.MODEL.CSL_HEAD.POOLER_RESOLUTION * 16
        segmentation_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        localisation_classes = cfg.MODEL.CSL_HEAD.LOCALISATION_CLASSES
        if self.hm_preprocessing_type == "align":
            self.csl_hm_align = ROIAlign((self.output_resolution, self.output_resolution),
                                         spatial_scale=1, sampling_ratio=0)
        elif self.hm_preprocessing_type == "direct":
            self.hm_direct_padding = 10
        self.decoder = nn.ModuleList([Decoder(localisation_classes).to(self.device) for i in range(segmentation_classes)])

    def _preprocess_hm_align(self, instances):
        calculated_heatmaps = {}
        heatmaps = []
        for keypoints_per_instance, box in zip(instances.gt_keypoints.keypoints, instances.proposal_boxes):
            heatmaps_per_instance = []
            for keypoints_per_class in keypoints_per_instance:
                kstring = str(keypoints_per_class)
                if kstring in calculated_heatmaps:
                    heatmap = calculated_heatmaps[kstring]
                else:
                    heatmap = np.zeros((instances.image_size[1], instances.image_size[0]))
                    if keypoints_per_class:
                        for x, y in keypoints_per_class:
                            heatmap[x, y] = 1
                        heatmap = max_gaussian_help(heatmap, self.sigma, 1)
                        heatmap = heatmap * 2 * np.pi * (self.sigma ** 2)  # normalize
                        calculated_heatmaps[kstring] = heatmap
                    else:
                        heatmap = np.expand_dims(heatmap, axis=2)
                heatmaps_per_instance.append(torch.from_numpy(heatmap.transpose((2, 1, 0))))
            # we align every heatmap tensor individually due to some weird bug resulting in a mismatch of
            # ground truth instances and its heatmaps when doing the alignment at the end, on a
            # stacked tensor of all heatmaps
            box = convert_boxes_to_pooler_format([Boxes(box.unsqueeze(0))]).cpu()
            heatmaps_per_instance = torch.cat(heatmaps_per_instance).unsqueeze(0).float()

            heatmaps.append(self.csl_hm_align(heatmaps_per_instance, box))
        return torch.cat(heatmaps)

    def _preprocess_hm_direct(self, instances):
        padded_size = self.output_resolution + 2 * self.hm_direct_padding
        heatmaps = []
        for keypoints_per_instance, box in zip(instances.gt_keypoints.keypoints, instances.proposal_boxes):
            w, h = box[2] - box[0], box[3] - box[1]
            ratio_h = (self.output_resolution / max(h, 0.1)).item()
            ratio_w = (self.output_resolution / max(w, 0.1)).item()
            heatmaps_per_instance = []
            for keypoints_per_class in keypoints_per_instance:
                heatmap = np.zeros((padded_size, padded_size))
                for keypoint in keypoints_per_class:
                    # fit keypoints to squared mask (if in box)
                    keypoint = (round((keypoint[0] - box[0].item()) * ratio_w) + self.hm_direct_padding,
                                round((keypoint[1] - box[1].item()) * ratio_h) + self.hm_direct_padding)
                    x, y = keypoint
                    if (0 <= x < padded_size) and (0 <= y < padded_size):
                        heatmap[keypoint[0], keypoint[1]] = 1

                heatmap = max_gaussian_help(heatmap, self.sigma, 1)
                heatmap = heatmap * 2 * np.pi * (self.sigma ** 2)  # normalize
                heatmap = heatmap[self.hm_direct_padding:padded_size-self.hm_direct_padding,
                                  self.hm_direct_padding:padded_size-self.hm_direct_padding]
                heatmaps_per_instance.append(torch.from_numpy(heatmap).squeeze())
            heatmaps_per_instance = torch.stack(heatmaps_per_instance).float()
            heatmaps.append(heatmaps_per_instance)

        return torch.stack(heatmaps)

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
        if self.hm_preprocessing_type == "align":
            return self._preprocess_hm_align(instances)
        elif self.hm_preprocessing_type == "direct":
            return self._preprocess_hm_direct(instances)
        else:
            raise ValueError

    def _forward_per_instances(self, x, instances):
        output = []
        x = [i.split(1, 0) for i in x]
        box = 0
        for instances_per_image in instances:
            segs = []
            locs = []
            classes = instances_per_image.gt_classes.tolist() if self.training \
                else instances_per_image.pred_classes.tolist()
            for cls in classes:
                features = [i[box] for i in x]
                box += 1
                seg, loc = self.decoder[cls](features)
                segs.append(seg)
                locs.append(loc)
            seg = torch.cat(segs)
            loc = torch.cat(locs)
            output.append((seg, loc))
        return output

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
                current_heatmaps = self._preprocess_gt_heatmaps(instances_per_image)
                gt_locs.append(current_heatmaps)
            gt_masks = torch.cat(gt_masks, dim=0)
            gt_locs = torch.cat(gt_locs, dim=0)

            return {"loss_seg": self._segmentation_loss(seg, gt_masks),
                    "loss_loc": self.lambdaa * self._localisation_loss(loc, gt_locs)}
        else:
            for instances_per_image, (seg, loc) in zip(instances, x):
                instances_per_image.pred_masks = torch.sigmoid(seg)
                instances_per_image.pred_loc = loc
            return instances


def build_csl_head(cfg):
    name = cfg.MODEL.CSL_HEAD.NAME
    return CSL_HEAD_REGISTRY.get(name)(cfg)

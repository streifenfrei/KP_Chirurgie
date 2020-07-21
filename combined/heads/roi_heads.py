from typing import Dict, List, Optional, Tuple, Union

import torch
from detectron2.config import configurable
from detectron2.layers import ROIAlign
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler, convert_boxes_to_pooler_format
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import ImageList, Instances
from torch import nn

from combined.heads.csl_head import build_csl_head
from combined.heads.csl_pooler import CSLPooler


@ROI_HEADS_REGISTRY.register()
class CSLROIHeads(StandardROIHeads):
    """
    The roi heads module consisting of the builtin box head and a csl head. The box head proposes bounding boxes used by
    the csl head to do segmentation and localisation of keypoints
    """
    @configurable
    def __init__(
            self,
            *,
            csl_in_features: List[str],
            csl_pooler: CSLPooler,
            csl_head: nn.Module,
            **kwargs
    ):
        # disable mask and keypoint heads of StandardROIHeads
        kwargs['mask_in_features'] = kwargs['keypoint_in_features'] = None
        super().__init__(**kwargs)
        self.csl_in_features = csl_in_features
        self.csl_pooler = csl_pooler
        self.csl_head = csl_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = StandardROIHeads.from_config(cfg, input_shape)
        ret.update(cls._init_csl_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_csl_head(cls, cfg, input_shape):
        in_features = cfg.MODEL.CSL_HEAD.IN_FEATURES
        scales = list(1.0 / input_shape[k].stride for k in in_features)
        scales.reverse()
        pooler_resolution = cfg.MODEL.CSL_HEAD.POOLER_RESOLUTION
        csl_pooler = CSLPooler(pooler_resolution, scales, 0)
        return {
            "csl_in_features": in_features,
            "csl_pooler": csl_pooler,
            "csl_head": build_csl_head(cfg)
        }

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        instances, losses = super().forward(images, features, proposals, targets)
        if self.training:
            losses.update(self._forward_csl(features, instances))
        return instances, losses

    def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        instances = super().forward_with_given_boxes(features, instances)
        instances = self._forward_csl(features, instances)
        return instances

    def _forward_csl(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.csl_in_features]
            mask_features = self.csl_pooler(features, proposal_boxes)
            return self.csl_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            features = [features[f] for f in self.csl_in_features]
            mask_features = self.csl_pooler(features, pred_boxes)
            return self.csl_head(mask_features, instances)

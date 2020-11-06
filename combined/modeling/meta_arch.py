from typing import Tuple, Optional, List, Any

import torch
import numpy as np
from detectron2.config import configurable
from torch import nn
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY, Backbone
from detectron2.layers import paste_masks_in_image
from detectron2.layers.mask_ops import BYTES_PER_FLOAT, GPU_MEM_LIMIT, _do_paste_mask

from evaluate import applyThreshold, non_max_suppression
from detectron2.utils.memory import retry_if_cuda_oom


@META_ARCH_REGISTRY.register()
class RCNNAndCSL(GeneralizedRCNN):
    """
    Combined meta architecture. Adds some postprocessing for the csl output
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            keypoint_limits: List[int] = None
    ):
        super().__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads,
                         pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)
        self.keypoint_limits = keypoint_limits

    @classmethod
    def from_config(cls, cfg):
        dic = super().from_config(cfg)
        dic["keypoint_limits"] = cfg.MODEL.KEYPOINT_LIMITS
        return dic

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        results = super().inference(batched_inputs, detected_instances, do_postprocess)
        if do_postprocess:
            return self._postprocess_csl(results)
        else:
            return results

    def _postprocess_csl(self, results):
        for results_per_image in results:
            instances_per_image = results_per_image["instances"]
            keypoints = []
            for heatmaps_per_instance, box in zip(instances_per_image.pred_loc.split(1, dim=0),
                                                  instances_per_image.pred_boxes):
                x1_box, y1_box, x2_box, y2_box = box.detach().cpu().numpy()
                width_box, height_box = x2_box - x1_box, y2_box - y1_box
                keypoints_per_instance = []
                for i, heatmap in enumerate(heatmaps_per_instance.split(1, dim=1)):
                    heatmap = heatmap[0, 0, :, :].detach().cpu().numpy()
                    width, height = heatmap.shape
                    # apply threshold to filter noise
                    heatmap_thres = applyThreshold(heatmap, 0.8)
                    # extract keypoints using non maximum suppression (in fixed size coordinate system)
                    xy_predict = non_max_suppression(np.float32(heatmap_thres),
                                                     None if self.keypoint_limits is None else self.keypoint_limits[i])
                    keypoints_per_class = []
                    for x_hm, y_hm in xy_predict:
                        # fit keypoints to the image's coordinate system
                        x_norm, y_norm = x_hm / width, y_hm / height
                        x, y = int(np.round(x1_box + (x_norm * width_box))), int(np.round(y1_box + (y_norm * height_box)))
                        keypoints_per_class.append((x, y))
                    keypoints_per_instance.append(keypoints_per_class)
                keypoints.append(keypoints_per_instance)
            instances_per_image.pred_loc = keypoints
            results_per_image["instances"] = instances_per_image
        return results


def paste_hms_in_image(heatmaps, boxes, image_shape):

    assert heatmaps.shape[-1] == heatmaps.shape[-2], "Only square mask predictions are supported"
    N = len(heatmaps)
    if N == 0:
        return heatmaps.new_empty((0,) + image_shape, dtype=heatmaps.dtype)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=heatmaps.dtype
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            heatmaps[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks
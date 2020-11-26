from typing import Tuple, Optional, List

import numpy as np
from detectron2.config import configurable
from torch import nn
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY, Backbone
from torch.autograd import profiler

from combined.profile.putil import profiling
from util.evaluate import apply_threshold, non_max_suppression


@META_ARCH_REGISTRY.register()
class RCNNAndCSL(GeneralizedRCNN):
    """
    Combined meta architecture. Adds some postprocessing for the csl output
    """

    @configurable
    def __init__(
            self,
            *,
            hm_threshold: float = 0.8,
            keypoint_limits: List[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hm_threshold = hm_threshold
        self.keypoint_limits = keypoint_limits
        self.backbone.forward = profiling(self.backbone.forward, "backbone")
        self.proposal_generator.forward = profiling(self.proposal_generator.forward, "proposal_generator")

    @classmethod
    def from_config(cls, cfg):
        dic = super().from_config(cfg)
        dic["hm_threshold"] = cfg.MODEL.POSTPROCESSING.HM_THRESHOLD
        dic["keypoint_limits"] = cfg.MODEL.POSTPROCESSING.KEYPOINT_LIMITS
        return dic

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        results = super().inference(batched_inputs, detected_instances, do_postprocess)
        if do_postprocess:
            with profiler.record_function("csl_postprocess"):
                return self._postprocess_csl(results)
        else:
            return results

    def _postprocess_csl(self, results):
        """
        Extracts the keypoints, given the predicted and aligned heatmaps
        Args:
            results: result dictionary containing the predicted instances
        Returns:
            the same dictionary with the extracted keypoints
        """
        for results_per_image in results:  # for each image in batch
            instances_per_image = results_per_image["instances"]
            if len(instances_per_image) > 0:  # if image is not empty
                keypoints = []
                for heatmaps_per_instance, box in zip(instances_per_image.pred_loc.split(1, dim=0),
                                                      instances_per_image.pred_boxes):  # for each instance in image
                    # retrieve the coordinates/size of the instance's box (in the original image coordinate system)
                    x1_box, y1_box, x2_box, y2_box = box.detach().cpu().numpy()
                    width_box, height_box = x2_box - x1_box, y2_box - y1_box
                    keypoints_per_instance = []
                    for i, heatmap in enumerate(heatmaps_per_instance.split(1, dim=1)):  # for each heatmap of instance
                        heatmap = heatmap[0, 0, :, :].detach().cpu().numpy()
                        width, height = heatmap.shape
                        heatmap_thres = apply_threshold(heatmap, self.hm_threshold)  # apply threshold to filter noise
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
                instances_per_image.pred_hm = instances_per_image.pred_loc
                instances_per_image.pred_loc = keypoints  # update instance's .pred_loc field with new keypoint list
                results_per_image["instances"] = instances_per_image
        return results
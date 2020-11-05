import numpy as np
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY

from evaluate import applyThreshold, non_max_suppression


@META_ARCH_REGISTRY.register()
class RCNNAndCSL(GeneralizedRCNN):
    """
    Combined meta architecture. Adds some postprocessing for the csl output
    """

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        results = super().inference(batched_inputs, detected_instances, do_postprocess)
        if do_postprocess:
            return RCNNAndCSL._postprocess_csl(results)
        else:
            return results

    @staticmethod
    def _postprocess_csl(results):
        for results_per_image in results:
            instances_per_image = results_per_image["instances"]
            keypoints = []
            for heatmaps_per_instance, box in zip(instances_per_image.pred_loc.split(1, dim=0),
                                                  instances_per_image.pred_boxes):
                x1_box, y1_box, x2_box, y2_box = box.detach().cpu().numpy()
                width_box, height_box = x2_box - x1_box, y2_box - y1_box
                keypoints_per_instance = []
                for heatmap in heatmaps_per_instance.split(1, dim=1):
                    heatmap = heatmap[0, 0, :, :].detach().cpu().numpy()
                    width, height = heatmap.shape
                    # apply threshold to filter noise
                    heatmap_thres = applyThreshold(heatmap, 0.8)
                    # extract keypoints using non maximum suppression (in fixed size coordinate system)
                    xy_predict = non_max_suppression(np.float32(heatmap_thres))
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

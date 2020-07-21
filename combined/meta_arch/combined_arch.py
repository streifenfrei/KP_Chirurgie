from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from evaluate import non_max_suppression


@META_ARCH_REGISTRY.register()
class RCNNAndCSL(GeneralizedRCNN):

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        results = super().inference(batched_inputs, detected_instances, do_postprocess)
        if do_postprocess:
            for result in results:
                instances = result["instances"]
                result["instances"] = RCNNAndCSL._postprocess_csl(instances)
        return results

    @staticmethod
    def _postprocess_csl(instances):
        loc = instances.pred_loc
        instances.pred_keypoints = RCNNAndCSL.heatmaps_to_keypoints(loc)
        return instances

    @staticmethod
    def heatmaps_to_keypoints(heatmaps):
        keypoints = []
        for heatmaps_per_image in heatmaps.split(1, 0):
            keypoints_per_image = []
            for heatmap_per_class in heatmaps_per_image.split(1, 1):
                heatmap_np = heatmap_per_class.squeeze().detach().cpu().numpy()
                keypoints_per_image.append(non_max_suppression(heatmap_np))
            keypoints.append(keypoints_per_image)
        return keypoints
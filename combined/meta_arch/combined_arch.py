import torch
from detectron2.layers import paste_masks_in_image
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY

from detectron2.utils.memory import retry_if_cuda_oom


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
        locs = []
        for loc_per_class in loc.split(1, 1):
            locs.append(
                retry_if_cuda_oom(paste_masks_in_image)(
                    loc_per_class[:, 0, :, :],  # N, 1, M, M
                    instances.pred_boxes,
                    instances.image_size,
                    threshold=0, ).unsqueeze(1)
            )

        instances.pred_loc = torch.cat(locs, dim=1)
        return instances

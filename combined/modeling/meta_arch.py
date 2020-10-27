import torch
from detectron2.layers import paste_masks_in_image
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.utils.memory import retry_if_cuda_oom


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

            # scale heatmaps
            heatmaps = []
            for heatmap_per_class in instances_per_image.pred_loc.split(1, dim=1):
                heatmap_per_class = retry_if_cuda_oom(paste_masks_in_image)(
                    heatmap_per_class[:, 0, :, :],  # N, 1, M, M
                    instances_per_image.pred_boxes,
                    instances_per_image.image_size,
                    threshold=0.999,
                )
                heatmaps.append(heatmap_per_class)
            instances_per_image.pred_loc = torch.stack(heatmaps, dim=1)

            # fancy debugging visualisation
            #import matplotlib
            #matplotlib.use("TkAgg")
            #import matplotlib.pyplot as plt
            #fig = plt.figure(figsize=(12, 3))
            #for index, hm in enumerate(torch.sum(instances_per_image.pred_loc.cpu(), dim=0).to(torch.bool).split(1, 0)):
            #    print(torch.max(hm))
            #    fig.add_subplot(2, 4, index + 1)
            #    plt.imshow(hm.squeeze().detach())
            #plt.show()

            results_per_image["instances"] = instances_per_image
        return results

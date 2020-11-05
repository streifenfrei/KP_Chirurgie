import torch
import numpy as np
from detectron2.layers import paste_masks_in_image
from detectron2.layers.mask_ops import BYTES_PER_FLOAT, GPU_MEM_LIMIT, _do_paste_mask
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
                heatmap_per_class = retry_if_cuda_oom(paste_hms_in_image)(
                    heatmap_per_class[:, 0, :, :],  # N, 1, M, M
                    instances_per_image.pred_boxes,
                    instances_per_image.image_size
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
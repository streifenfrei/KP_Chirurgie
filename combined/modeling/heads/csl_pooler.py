from typing import List

import torch
from detectron2.layers import ROIAlign
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from torch import nn


class CSLPooler(nn.Module):
    """
    Pooler for generating aligned feature masks as input for the csl head
    """
    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
    ):

        super().__init__()
        # double the size of each feature mask (so they fit in the csl decoder)
        # e.g. 14, 28, 56, 112
        self.output_sizes = [output_size * (2 ** x) for x in range(4)]

        self.level_poolers = nn.ModuleList(
            ROIAlign(
                (size, size), spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
            )
            for size, scale in zip(self.output_sizes, scales)
        )

    def forward(self, x: List[torch.Tensor], box_lists):
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        x.reverse()  # we get the feature maps in descending order, so we have to reverse them to fit the decoder
        outputs = []
        for x_level, pooler in zip(x, self.level_poolers):
            output = pooler(x_level, pooler_fmt_boxes)
            outputs.append(output)

        return outputs

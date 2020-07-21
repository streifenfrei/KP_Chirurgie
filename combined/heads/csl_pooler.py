from typing import List

import torch
from detectron2.layers import ROIAlign
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from torch import nn


class CSLPooler(nn.Module):

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
    ):

        super().__init__()

        self.output_sizes = [output_size * (2 ** x) for x in range(4)]

        self.level_poolers = nn.ModuleList(
            ROIAlign(
                (size, size), spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
            )
            for size, scale in zip(self.output_sizes, scales)
        )

    def forward(self, x: List[torch.Tensor], box_lists):
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        x.reverse()
        outputs = []
        for x_level, pooler in zip(x, self.level_poolers):
            output = pooler(x_level, pooler_fmt_boxes)
            outputs.append(output)

        return outputs

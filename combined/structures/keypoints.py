from typing import Union
import torch


class CSLKeypoints:
    """
    Wrapper class for csl keypoints list which allows indexing with tensors
    like detectrons builtin structures (e.g. Keypoints, Boxes, PolygonMask...)
    (for example needed by ROIHeads.label_and_sample_proposals(...))
    """
    def __init__(self, keypoints):
        self.keypoints = keypoints

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]):
        if isinstance(item, int) or isinstance(item, slice):
            return self.keypoints[item]
        new_keypoints = []
        for index in item.split(1, 0):
            new_keypoints.append(self.keypoints[index])
        return CSLKeypoints(new_keypoints)

    def __len__(self):
        return len(self.keypoints)

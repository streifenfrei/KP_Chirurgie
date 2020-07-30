import copy

import torch
from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import apply_transform_gens, StandardAugInput
from detectron2.engine import DefaultTrainer

from combined.structures.keypoints import CSLKeypoints
import numpy as np


class Mapper(DatasetMapper):
    """
    Custom mapper which applies the transforms of DatasetMapper to our CSL keypoints
    """
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        if is_train:
            # remove flip transform, because I couldn't figure out how to create hflip indices for the csl keypoints :(
            # (detection_utils.create_keypoint_hflip_indices(...) only works with the COCO keypoints)
            self.augmentations.pop(-1)

    def __call__(self, dataset_dict):

        # COPIED FROM super method

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = StandardAugInput(image, sem_seg=None)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            # APPLYING TRANSFORMS TO CSL KEYPOINTS

            keypoints = []
            for instance in annos:
                keypoints_per_instance = []
                for keypoints_per_class in instance["keypoints_csl"].values():
                    keypoints_per_class = np.asarray(keypoints_per_class, dtype="float64").reshape(-1, 2)
                    keypoints_per_class = transforms.apply_coords(keypoints_per_class).tolist()
                    keypoints_per_class = [(round(k[0]), round(k[1])) for k in keypoints_per_class]
                    keypoints_per_instance.append(keypoints_per_class)
                keypoints.append(keypoints_per_instance)

            instances.gt_keypoints = CSLKeypoints(keypoints)
            dataset_dict["instances"] = instances

        return dataset_dict


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=Mapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=Mapper(cfg, True))

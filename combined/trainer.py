import copy

import torch
from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import apply_transform_gens, ResizeShortestEdge
from detectron2.engine import DefaultTrainer

from combined.structures.keypoints import CSLKeypoints
import numpy as np


class Mapper:
    """
    Custom mapper which applies the transforms of DatasetMapper to our CSL keypoints
    """
    def __init__(self, cfg, is_train=True):
        self.augmentations = []
        self.img_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.is_train = is_train
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        self.augmentations.append(ResizeShortestEdge(min_size, max_size, sample_style))

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = apply_transform_gens(self.augmentations, image)

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:

            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

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

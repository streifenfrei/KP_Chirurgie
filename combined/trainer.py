import copy

import torch
from dateutil import utils
from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper, detection_utils
from detectron2.data.transforms import apply_transform_gens, HFlipTransform
from detectron2.engine import DefaultTrainer
from detectron2.layers import paste_masks_in_image
from detectron2.structures import Boxes

from combined.meta_arch.keypoints import CSLKeypoints
import numpy as np

from dataLoader import max_gaussian_help


class Mapper(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        if is_train:
            self.tfm_gens.pop(-1)

    def __call__(self, dataset_dict):
        #out_dict = super().__call__(dataset_dict)

        # COPIED FROM DatasetMapper

        dataset_dict = copy.deepcopy(dataset_dict)
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.img_format)
        detection_utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            if self.crop_gen:
                crop_tfm = detection_utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                detection_utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = detection_utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            instances = detection_utils.filter_empty_instances(instances)

            scale_x = instances.image_size[1] / dataset_dict["width"]
            scale_y = instances.image_size[0] / dataset_dict["height"]

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
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=Mapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=Mapper(cfg, True))

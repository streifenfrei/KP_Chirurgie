from argparse import ArgumentParser

import torch, torchvision
import cython
import detectron2
from detectron2.utils.logger import setup_logger, log_every_n
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import random
import cv2
from typing import List, Tuple
import math

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

# import some common detectron2 utilities
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.config import CfgNode
from pprint import pprint
import os
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.utils.events import get_event_storage

# from maskrcnn.custom_dataloader import mapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader
import copy
# inside the model:

landmark_name_to_id_ = {
'jaw':1,
'center':2,
'joint':3,
'shaft':4
}


def inference_old_model(image_path: str = "../dataset/frame_00000.png") -> None:
    """
    This is an example inference based on a image that will be provided

    Returns:
        Nothing to return, only shows the inference results via cv2.imshow
    """
    image_path = "/Users/chernykh_alexander/Yandex.Disk.localized/CloudTUD/Komp_CHRIRURGIE/instruments/train/frame_00000_0.png"
    im = cv2.imread(image_path)

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_4000000.pth")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow('image', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


def get_balloon_dicts(img_dir: str,
                      json_with_desription_name: str = "dataset_registration_detectron2.json") -> List[dict]:
    """
    Creating a description for each image in the image dir according to the json description
    While extracting the description of from the json, Bounding Boxes will be also calculated

    Args:
        img_dir: dir where the images are located
        json_with_desription_name: description json of every single image in the img_dir

    Returns:
        List with description of every image
        F.e
        [{'height': 540, 'width': 960, 'file_name':
        '/instruments/train/frame_00000.png',
         'image_id': 0, 'annotations': [{'bbox': [0.0, 218.12345679012344, 630.0987654320987, 539.0],
         'bbox_mode': <BoxMode.XYXY_ABS: 0>, 'segmentation': [[0.5, 521.5, 435.537037037037, 297.6358024691358,
            , 355.2901234567901, 415.537037037037, 125.5, 539.5, 1.5864197530864197, 538.9938271604938]],
            'category_id': 2}, ..... ]


    """
    json_file = os.path.join(img_dir, json_with_desription_name)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        record["height"] = v['height']
        record["width"] = v['width']
        record["file_name"] = filename
        record["image_id"] = idx

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]


            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": anno['category_id'],
                "keypoints_csl" : anno['keypoints_csl']
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# if __name__ == '__main__':
def register_dataset_and_metadata(path_to_data: str,
                                  classes_list: List[str]) -> detectron2.data.catalog.Metadata:
    """
    Registrs the dataset according to the https://detectron2.readthedocs.io/tutorials/datasets.html

    Args:
        path_to_data: path to the folder, where the train and validation forlder is located
                      folder train has images for training and a json that describes the
                      data (bounding boxes, labels etc)
        classes_list: is a list of all possible labels that might occur

    Returns:
        a registration Metadata object that can be further used for training/testing/validation
        it is similar to a Dataloader

    """

    # classes_list = ['scissors', 'needle_holder', 'grasper']
    # path_to_data = "/Users/chernykh_alexander/Yandex.Disk.localized/CloudTUD/Komp_CHRIRURGIE/instruments/"
    for d in ["train", "val"]:
        DatasetCatalog.register("instruments_" + d, lambda d=d: get_balloon_dicts(path_to_data + d))
        MetadataCatalog.get("instruments_" + d).set(thing_classes=classes_list)
    instruments_metadata = MetadataCatalog.get("instruments_train")
    # instruments_metadata_val = MetadataCatalog.get("instruments_val")
    return instruments_metadata


def test_registration(instruments_metadata: detectron2.data.catalog.Metadata,
                      path_to_training_data: str,
                      json_with_desription_name: str = "dataset_registration_detectron2.json") -> None:
    """
    testing the registred dataset and its metadata by visualising the results of the annotation on the image


    Args:
        instruments_metadata: the registred data
        path_to_training_data:

    Returns:

    """
    dataset_dicts = get_balloon_dicts(path_to_training_data,
                                      json_with_desription_name=json_with_desription_name)
    for d in random.sample(dataset_dicts, 15):
        img = cv2.imread(d["file_name"])
        print(f'Took: {d["file_name"]}')
        visualizer = Visualizer(img[:, :, ::-1], metadata=instruments_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('image', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def load_config(config_path: str = None):
    assert config_path
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(config_path)
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
     #   "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints_csl" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints_csl"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints_csl"] = keypoints

    return annotation


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks


    #todo: add here the keypoints processing
    if len(annos) and "keypoints_csl" in annos[0]:
        kpts = [obj.get("keypoints_csl", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target


def mapper(dataset_dict):
    # Here we implement a minimal mapper for instance detection/segmentation

    dataset_dict = copy.deepcopy(dataset_dict)# it will be modified by code below
    dataset_dict_copy = copy.copy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    #
    # dataset_dict["image"] = torch.from_numpy(image.transpose(2, 0, 1))
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    transform_list = [T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
                      T.RandomFlip(prob=0.6, horizontal=False, vertical=True)]

    #todo: use the transofrmations for the keypoints

    # image, transforms = T.apply_transform_gens(transform_list, image)
    # annos = [
    #     transform_instance_annotations(obj, transforms, image.shape[:2])
    #     for obj in dataset_dict.pop("annotations")
    # ]

    dataset_dict["instances"] = annotations_to_instances(dataset_dict['annotations'], image.shape[:2])
    # TODO: call here load_pose from Xi
    dataset_dict['instances']._fields['gt_jaw']= torch.ones(2)

    image = dataset_dict["image"].numpy()
    seg_image = np.zeros_like(image, dtype=int)

    # create a imagessize zero array and put on a position of coordinates a certain value accroding to encoding
    # landmark_name_to_id_
    # seg_image = max_gaussian_help(cls, pose_sigma, 1)
    # normalize_heatmap == True:
    # for each_class in landmark_name_to_id:
    #     if each_class != 'jaw':
    #         seg_image_cls = max_gaussian_help(cls, pose_sigma, landmark_name_to_id[each_class])
    #         seg_image = np.dstack((seg_image, seg_image_cls))
    # if normalize_heatmap == True:
    #     seg_image = seg_image * 2 * np.pi * pose_sigma * pose_sigma
    # else:
    #     seg_image_cls = seg_image_cls * np.sqrt(2 * np.pi) * pose_sigma


    # for point in jaw:
        # position is to 1

    # dataset_dict['jaw'] = []
    # dataset_dict['shaft'] = []
    # dataset_dict['center'] = []
    # dataset_dict['joint'] =[]
    return dataset_dict


class Trainer(DefaultTrainer):
    # @classmethod
    # def build_evaluator(cls, cfg: CfgNode, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
    #     if cfg.MODEL.DENSEPOSE_ON:
    #         evaluators.append(DensePoseCOCOEvaluator(dataset_name, True, output_folder))
    #     return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=mapper)

    # @classmethod
    # def test_with_TTA(cls, cfg: CfgNode, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     # In the end of training, run an evaluation with TTA
    #     # Only support some R-CNN models.
    #     logger.info("Running inference with test-time augmentation ...")
    #     transform_data = load_from_cfg(cfg)
    #     model = DensePoseGeneralizedRCNNWithTTA(
    #         cfg, model, transform_data, DensePoseDatasetMapperTTA(cfg)
    #     )
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #         )
    #         for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #     return res








def start_training(cfg):

    # trainer = DefaultTrainer(cfg)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


def inference_on_trained_mode(instruments_metadata,
                              path_to_data,
                              cfg) -> None:
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_balloon_dicts(f"{path_to_data}/val")
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=instruments_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('yay.png', out.get_image()[:, :, ::-1])


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, default='configs/pretrained.yaml')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')

    args = arg_parser.parse_args()
    cfg = load_config(config_path=args.config)

    # saving all logs
    setup_logger(os.path.join(cfg.OUTPUT_DIR, 'saved_logs.log'))

    classes_list = ['scissors', 'needle_holder', 'grasper']
    instruments_metadata = register_dataset_and_metadata(args.dataset, classes_list)
    # path_to_val_data = "/Users/chernykh_alexander/Yandex.Disk.localized/CloudTUD/Komp_CHRIRURGIE/instruments/val"
    # test_registration(instruments_metadata, path_to_val_data, json_with_desription_name="dataset_registration_detectron2.json")

    # inference_old_model()
    start_training(cfg)
    #inference_on_trained_mode(instruments_metadata, args.dataset, cfg=cfg)


if __name__ == "__main__":
    main()

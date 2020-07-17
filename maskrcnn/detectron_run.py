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
from detectron2.data import build_detection_train_loader
import copy
# inside the model:


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
                "keypoints" : anno['keypoints']
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
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


def mapper(dataset_dict):
    # Here we implement a minimal mapper for instance detection/segmentation
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
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
        cv2.imshow('image', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


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
    inference_on_trained_mode(instruments_metadata, args.dataset, cfg=cfg)


if __name__ == "__main__":
    main()

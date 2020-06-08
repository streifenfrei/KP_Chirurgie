import torch, torchvision
import cython
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import random
import cv2
from typing import List

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from pprint import pprint
import os
import numpy as np
import json
from detectron2.structures import BoxMode


def inference_old_model(image_path: str = "../dataset/frame_00000.png") -> None:
    """
    This is an example inference based on a image that will be provided

    Returns:
        Nothing to return, only shows the inference results via cv2.imshow
    """
    im = cv2.imread(image_path)

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow('image', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


def get_balloon_dicts(img_dir:str,
                      json_with_desription_name: str = "dataset_registration_detectron2.json")->List[dict]:
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
                "category_id": anno['label'],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset_and_metadata(path_to_data, classes_list: List[str]) -> detectron2.data.catalog.Metadata:
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
    for d in ["train", "val"]:
        DatasetCatalog.register("instruments_" + d, lambda d=d: get_balloon_dicts(path_to_data + d))
        MetadataCatalog.get("instruments_" + d).set(thing_classes=classes_list)
    instruments_metadata = MetadataCatalog.get("instruments_train")
    return instruments_metadata


def test_registration(instruments_metadata: detectron2.data.catalog.Metadata, path_to_training_data: str,
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
        visualizer = Visualizer(img[:, :, ::-1], metadata=instruments_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('image', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)



def start_training(train_name:str="instruments_train", classes_list:List[str]=['scissors', 'needle_holder', 'grasper']):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes_list)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    classes_list = ['scissors', 'needle_holder', 'grasper']
    path_to_data = "/Users/chernykh_alexander/Yandex.Disk.localized/CloudTUD/Komp_CHRIRURGIE/instruments/"
    instruments_metadata = register_dataset_and_metadata(path_to_data, classes_list)
    path_to_training_data = "/Users/chernykh_alexander/Yandex.Disk.localized/CloudTUD/Komp_CHRIRURGIE/instruments/train"

    # test_registration(instruments_metadata, path_to_training_data,
    #                   json_with_desription_name="dataset_registration_detectron2.json")

    start_training(train_name="instruments_train", classes_list=['scissors', 'needle_holder', 'grasper'])



if __name__ == "__main__":
    main()

from argparse import ArgumentParser

import combined.heads.roi_heads, combined.heads.csl_head, combined.meta_arch.combined_arch
import torch
import detectron2
from detectron2.utils.logger import setup_logger
import random
import cv2
from typing import List

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

import os
import numpy as np
import json
from detectron2.structures import BoxMode

from combined.configs.config import add_csl_config


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
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


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

    for d in ["train", "val"]:
        DatasetCatalog.register("instruments_" + d, lambda d=d: get_balloon_dicts(path_to_data + d))
        MetadataCatalog.get("instruments_" + d).set(thing_classes=classes_list)
    instruments_metadata = MetadataCatalog.get("instruments_train")
    return instruments_metadata


def load_config(config_path: str = None):
    assert config_path
    cfg = get_cfg()
    add_csl_config(cfg)
    cfg.merge_from_file(config_path)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def start_training(cfg):
    trainer = DefaultTrainer(cfg)
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
    # inference_old_model()
    #start_training(cfg)
    inference_on_trained_mode(instruments_metadata, args.dataset, cfg=cfg)


if __name__ == "__main__":
    main()
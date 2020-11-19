import os
import random
from argparse import ArgumentParser

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode

from combined.configs.config import *
from combined.trainer import Trainer
from combined.visualizer import CSLVisualizer
from detectron2_commons.commons import register_dataset_and_metadata, get_instrument_dicts


def load_config(config_path: str = None):
    assert config_path
    cfg = get_cfg()
    # add new csl entries to the config dict, so we don't get an error when parsing them from the config file
    add_csl_config(cfg)
    cfg.merge_from_file(config_path)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def training(cfg):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


def inference(metadata,
              path_to_data,
              cfg) -> None:
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DefaultPredictor(cfg)
    dataset_dicts = get_instrument_dicts(f"{path_to_data}/val")
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = CSLVisualizer(im[:, :, ::-1],
                          metadata=metadata,
                          scale=0.8,
                          instance_mode=ColorMode.SEGMENTATION
                          )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, 'yay.png'), out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, default='configs/default.yaml')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')
    arg_parser.add_argument("--train", "-t", action="store_true")
    args = arg_parser.parse_args()
    cfg = load_config(config_path=args.config)
    setup_logger(os.path.join(cfg.OUTPUT_DIR, 'saved_logs.log'))
    metadata = register_dataset_and_metadata(args.dataset, cfg.VISUALIZER.CLASS_NAMES)
    metadata.set(keypoint_colors=cfg.VISUALIZER.KEYPOINT_COLORS)
    metadata.set(thing_colors=cfg.VISUALIZER.INSTANCE_COLORS)
    if args.train:
        training(cfg)
    else:
        inference(metadata, args.dataset, cfg=cfg)

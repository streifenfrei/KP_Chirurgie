import os
import random
from argparse import ArgumentParser

import cv2
from detectron2.config import get_cfg
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode

from combined.configs.config import *
from combined.trainer import Trainer
from combined.visualizer import CSLVisualizer
from maskrcnn.detectron_run import register_dataset_and_metadata, get_balloon_dicts


def load_config(config_path: str = None):
    assert config_path
    cfg = get_cfg()
    add_csl_config(cfg)
    cfg.merge_from_file(config_path)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def start_training(cfg):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def inference_on_trained_model(instruments_metadata,
                               path_to_data,
                               cfg) -> None:
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_balloon_dicts(f"{path_to_data}/val")
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = CSLVisualizer(im[:, :, ::-1],
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
    # inference_old_model()
    start_training(cfg)
    #inference_on_trained_model(instruments_metadata, args.dataset, cfg=cfg)


if __name__ == "__main__":
    main()

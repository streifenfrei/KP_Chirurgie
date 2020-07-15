import os
from argparse import ArgumentParser

from detectron2.config import get_cfg
# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

from combined.configs.config import *
from maskrcnn.detectron_run import register_dataset_and_metadata, inference_on_trained_mode


def load_config(config_path: str = None):
    assert config_path
    cfg = get_cfg()
    add_csl_config(cfg)
    cfg.merge_from_file(config_path)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def start_training(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


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
    #inference_on_trained_mode(instruments_metadata, args.dataset, cfg=cfg)


if __name__ == "__main__":
    main()

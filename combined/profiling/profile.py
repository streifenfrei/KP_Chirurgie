import os
from argparse import ArgumentParser

import cv2
import torch.autograd.profiler as profiler
from detectron2.engine import DefaultPredictor

from combined.combined_run import load_config
from detectron2_commons.commons import register_dataset_and_metadata, get_instrument_dicts


def inference(path_to_data, cfg):
    register_dataset_and_metadata(args.dataset, cfg.VISUALIZER.CLASS_NAMES)
    predictor = DefaultPredictor(cfg)
    dataset_dict = get_instrument_dicts(f"{path_to_data}/val")[0]
    im = cv2.imread(dataset_dict["file_name"])

    # profiling
    with profiler.profile(record_shapes=True) as prof:
        out = predictor(im)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, default='configs/default.yaml')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')
    args = arg_parser.parse_args()
    cfg = load_config(config_path=args.config)
    inference(args.dataset, cfg)


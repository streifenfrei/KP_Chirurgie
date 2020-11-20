from argparse import ArgumentParser

import torch.autograd.profiler as profiler

from combined.combined_run import load_config, inference
from detectron2_commons.commons import register_dataset_and_metadata


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, default='configs/default.yaml')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')
    args = arg_parser.parse_args()
    cfg = load_config(config_path=args.config)
    metadata = register_dataset_and_metadata(args.dataset, cfg.VISUALIZER.CLASS_NAMES)
    metadata.set(keypoint_colors=cfg.VISUALIZER.KEYPOINT_COLORS)
    metadata.set(thing_colors=cfg.VISUALIZER.INSTANCE_COLORS)
    with profiler.profile(record_shapes=True) as prof:
        inference(metadata, args.dataset, cfg=cfg)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")

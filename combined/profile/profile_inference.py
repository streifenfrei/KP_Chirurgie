import platform
import shutil

import cProfile
import os
from argparse import ArgumentParser
from pstats import SortKey

import cv2
import psutil as psutil
import torch
import torch.autograd.profiler as profiler
from detectron2.engine import DefaultPredictor

from combined.combined_run import load_config
from combined.trainer import Trainer
from detectron2_commons.commons import register_dataset_and_metadata, get_instrument_dicts

from pyprof2calltree import convert
import json


def inference(path_to_data, cfg, output):
    output_root = os.path.join(output, "profile")
    if os.path.exists(output_root) and os.path.isdir(output_root):
        raise FileExistsError("profile directory already exists in output directory")
    os.mkdir(output_root)
    try:
        register_dataset_and_metadata(args.dataset, cfg.VISUALIZER.CLASS_NAMES)
        dataset_dict = get_instrument_dicts(f"{path_to_data}/val")[0]
        im = cv2.imread(dataset_dict["file_name"])

        with open(os.path.join(output_root, 'system_specs.json'), 'w') as spec_file:
            info = {}
            uname = platform.uname()
            info["System"] = uname.system
            info["Release"] = uname.release
            info["Version"] = uname.version
            info["Machine"] = uname.machine
            info["Processor"] = uname.processor
            info['RAM'] = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
            spec_file.write(json.dumps(info, indent=4))

        # EVALUATION
        with open(os.path.join(output_root, 'evaluation.json'), 'w') as eval_file:
            model = Trainer.build_model(cfg)
            model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
            eval_file.write(json.dumps(Trainer.test(cfg, model), indent=4))
        return
        # PROFILING
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            output = os.path.join(output_root, device)
            os.mkdir(output)
            cfg.MODEL.DEVICE = device
            predictor = DefaultPredictor(cfg)
            predictor(im)  # warm up model (construct indices, memory allocator...)
            # pytorch profiler

            with profiler.profile(record_shapes=True) as prof:
                predictor(im)
            results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=100)
            with open(os.path.join(output, "torch_profile.txt"), "w") as file:
                file.write(results)
            print(results)
            prof.export_chrome_trace(os.path.join(output, "torch_trace.json"))
            # cProfile
            with cProfile.Profile() as prof:
                predictor(im)
            prof.print_stats(sort=SortKey.TIME)
            cprofile_file = os.path.join(output, "cprofile.pstats")
            prof.dump_stats(file=cprofile_file)
            # gprof2dot
            os.system("gprof2dot -f pstats '{0}' | dot -Tpng -o '{1}'".format(cprofile_file, os.path.join(output, "calltree.png")))
            # pyprof2calltree
            convert(prof.getstats(), os.path.join(output, "calltree.kgrind"))
    except Exception as e:
        shutil.rmtree(output_root)
        raise e


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, default='configs/default.yaml')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')
    arg_parser.add_argument("--output", "-o", type=str)
    args = arg_parser.parse_args()
    cfg = load_config(config_path=args.config)
    output = args.output if args.output is not None else cfg.OUTPUT_DIR
    inference(args.dataset, cfg, output)


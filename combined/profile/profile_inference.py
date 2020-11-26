import shutil

import cProfile
import os
from argparse import ArgumentParser
from pstats import SortKey

import cv2
import torch.autograd.profiler as profiler
from detectron2.engine import DefaultPredictor

from combined.combined_run import load_config
from detectron2_commons.commons import register_dataset_and_metadata, get_instrument_dicts

from pyprof2calltree import convert


def inference(path_to_data, cfg, output):
    output_root = os.path.join(output, "profile")
    if os.path.exists(output_root) and os.path.isdir(output_root):
        raise FileExistsError("profile directory already exists in output directory")
    os.mkdir(output_root)
    try:
        register_dataset_and_metadata(args.dataset, cfg.VISUALIZER.CLASS_NAMES)
        dataset_dict = get_instrument_dicts(f"{path_to_data}/val")[0]
        im = cv2.imread(dataset_dict["file_name"])

        # TODO EVALUATOR
        #model = Trainer.build_model(cfg)
        #res = Trainer.test(cfg, model)

        # PROFILING
        for device in ["cpu", "cuda"]:
            output = os.path.join(output_root, device)
            os.mkdir(output)
            cfg.MODEL.DEVICE = device
            predictor = DefaultPredictor(cfg)
            predictor(im)  # warm up model (construct indices, memory allocator...)
            # pytorch profiler

            with profiler.profile(record_shapes=True) as prof:
                predictor(im)
            results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
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
            os.system("gprof2dot -f pstats {0} | dot -Tpng -o {1}".format(cprofile_file, os.path.join(output, "calltree.png")))
            # pyprof2calltree
            convert(prof.getstats(), os.path.join(output, "calltree.kgrind"))
    except Exception as e:
        shutil.rmtree(output)
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


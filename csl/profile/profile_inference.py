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
from csl.csl_run import _load_model

import numpy as np
from pyprof2calltree import convert
import json

from csl.data_loader import image_norm, image_transform_valid


def inference(input, weights, output):
    output_root = os.path.join(output, "profile")
    if os.path.exists(output_root) and os.path.isdir(output_root):
        raise FileExistsError("profile directory already exists in output directory")
    os.mkdir(output_root)
    try:
        checkpoint = torch.load(weights)
        model, _ = _load_model(checkpoint['model_state_dict'])
        model.eval()
        input = cv2.imread(input)
        input = image_norm()(image_transform_valid(p=1)(image=input)["image"])
        input = torch.unsqueeze(input, 0)
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

        # PROFILING
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            output = os.path.join(output_root, device)
            os.mkdir(output)
            current_model = model.to(device)
            current_input = input.to(device)
            current_model(current_input)  # warm up model (construct indices, memory allocator...)
            # pytorch profiler

            with profiler.profile(record_shapes=True) as prof:
                current_model(current_input)
            results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=100)
            with open(os.path.join(output, "torch_profile.txt"), "w") as file:
                file.write(results)
            print(results)
            prof.export_chrome_trace(os.path.join(output, "torch_trace.json"))
            # cProfile
            with cProfile.Profile() as prof:
                current_model(current_input)
            prof.print_stats(sort=SortKey.TIME)
            cprofile_file = os.path.join(output, "cprofile.pstats")
            prof.dump_stats(file=cprofile_file)
            # gprof2dot
            os.system("gprof2dot -f pstats '{0}' | dot -Tpng -o '{1}'".format(cprofile_file, os.path.join(output, "calltree.png")))
            # pyprof2calltree
            convert(prof.getstats(), os.path.join(output, "calltree.kgrind"))
    except (Exception, StopIteration) as e:
        shutil.rmtree(output_root)
        raise e


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", "-i", type=str, required=True)
    arg_parser.add_argument("--weights", "-w", type=str, required=True)
    arg_parser.add_argument("--output", "-o", type=str, required=True)
    args = arg_parser.parse_args()
    inference(args.input, args.weights, args.output)


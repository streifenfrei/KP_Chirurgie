from argparse import ArgumentParser

import cv2
import torch

from csl.data_loader import image_norm, image_transform_valid
from csl.quantization.model import QuantizableCSLNet

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", "-i", type=str, required=True)
    arg_parser.add_argument("--weights", "-w", type=str, required=True)
    args = arg_parser.parse_args()
    input = cv2.imread(args.input)
    input = image_norm()(image_transform_valid(p=1)(image=input)["image"])
    input = torch.unsqueeze(input, 0)
    checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
    model = QuantizableCSLNet(input_shape=input.shape)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model(input)  # warm up
    torch.jit.trace(model, input)
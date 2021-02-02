from argparse import ArgumentParser

import torch
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from csl.net import CSLNet

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--weights", "-w", type=str, required=True)
    arg_parser.add_argument("--batch_size", "-b", type=int, default=1)
    arg_parser.add_argument("--quant_mode", "-q", choices=["calib", "test"], required=True)
    arg_parser.add_argument("--deploy", "-d", action="store_true")
    arg_parser.add_argument("--finetune", "-ft", action="store_true")
    args = arg_parser.parse_args()
    input = torch.randn((args.batch_size, 3, 512, 960))
    checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
    model = CSLNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    quantizer = torch_quantizer(args.quant_mode, model, (input))
    model = quantizer.quant_model
    model(input)
    if args.quant_mode == "calib":
        if args.finetune:
            pass
        quantizer.export_quant_config()

    elif args.quant_mode == "test":
        if args.finetune:
            quantizer.load_ft_param()
        if args.deploy:
            quantizer.export_xmodel(deploy_check=False)



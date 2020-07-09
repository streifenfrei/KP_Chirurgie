import os
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter


def convert(workspace):
    csv_file_path = os.path.join(workspace, 'csl_val.csv')
    writer = SummaryWriter(log_dir=os.path.join(workspace, 'tensorboard'))
    with open(csv_file_path, 'r') as csv_file:
        for line in csv_file:
            line = line.rstrip("\n")
            if line:
                line_list = line.split(',')
                epoch = int(line_list[0])
                values = list(map(float, line_list[1:]))
                average_loss = sum(values) / len(values)
                writer.add_scalar("Loss/validation", average_loss, epoch)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--workspace", "-w", type=str, default='.')
    args = arg_parser.parse_args()
    convert(args.workspace)

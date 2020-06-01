import os
from argparse import ArgumentParser

import torch
from csl.net import CSLNet, train
from dataLoader import image_transform, OurDataLoader

segmentation_classes = 3
localisation_classes = 4
learning_rate = 0.01
sigma = 5


def init_model(save_file):
    model = CSLNet(segmentation_classes=3,
                   localisation_classes=4)
    model.load_state_dict(torch.load(os.path.abspath("weights/resnet50-19c8e357.pth")), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0,
    }, save_file)


def train_model(workspace, dataset, normalize_heatmap=False):
    model = CSLNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint = torch.load(os.path.join(workspace, 'csl.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    dataset = OurDataLoader(data_dir=dataset, task_type='both', transform=image_transform(p=1), pose_sigma=sigma,
                            normalize_heatmap=normalize_heatmap)
    epoch = checkpoint['epoch']

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'gpu'
    train(model, dataset, optimizer, start_epoch=epoch, output=workspace, device=device)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--init", "-i", action='store_true', default=False)
    arg_parser.add_argument("--workspace", "-w", type=str, default='.')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')
    arg_parser.add_argument("--normalize", "-n", action='store_true', default=False)

    args = arg_parser.parse_args()
    if args.init:
        model_out = os.path.join(args.workspace, 'csl.pth')
        init_model(model_out)
    else:
        train_model(args.workspace, args.dataset, args.normalize)

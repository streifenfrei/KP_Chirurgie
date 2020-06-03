import os
from argparse import ArgumentParser

import torch
from csl.net import CSLNet, train, visualize
from dataLoader import image_transform, OurDataLoader

segmentation_classes = 4
localisation_classes = 4
learning_rate = 0.01
sigma = 5


def init_model(save_file):
    model = CSLNet(segmentation_classes=segmentation_classes,
                   localisation_classes=localisation_classes)
    model.load_state_dict(torch.load(os.path.abspath("weights/resnet50-19c8e357.pth")), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0,
    }, save_file)


def _load_model(state_dict):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = CSLNet(segmentation_classes=segmentation_classes,
                   localisation_classes=localisation_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, device


def train_model(workspace, dataset, normalize_heatmap=False):
    checkpoint = torch.load(os.path.join(workspace, 'csl.pth'))
    model, device = _load_model(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    dataset = OurDataLoader(data_dir=dataset, task_type='both', transform=image_transform(p=1),
                            pose_sigma=sigma,
                            normalize_heatmap=normalize_heatmap)
    epoch = checkpoint['epoch']
    if device == 'cuda':
        del checkpoint
        torch.cuda.empty_cache()
    train(model, dataset, optimizer, start_epoch=epoch, workspace=workspace, device=device)


def call_model(workspace, dataset, normalize_heatmap=False):
    checkpoint = torch.load(os.path.join(workspace, 'csl.pth'))
    model, device = _load_model(checkpoint['model_state_dict'])
    dataset = OurDataLoader(data_dir=dataset, task_type='both', transform=image_transform(p=1),
                            pose_sigma=sigma,
                            normalize_heatmap=normalize_heatmap)
    if device == 'cuda':
        del checkpoint
        torch.cuda.empty_cache()
    visualize(model, dataset, device)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--command", "-c", type=str, choices=['init', 'train', 'call'], default='train')
    arg_parser.add_argument("--workspace", "-w", type=str, default='.')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')
    arg_parser.add_argument("--normalize", "-n", action='store_true', default=False)

    args = arg_parser.parse_args()
    if args.command == 'init':
        model_out = os.path.join(args.workspace, 'csl.pth')
        init_model(model_out)
    elif args.command == 'train':
        train_model(args.workspace, args.dataset, args.normalize)
    elif args.command == 'call':
        call_model(args.workspace, args.dataset)

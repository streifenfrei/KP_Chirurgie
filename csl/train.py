import os
from argparse import ArgumentParser

import torch
from csl.net import CSLNet, Training
from torch.optim.lr_scheduler import *
from dataLoader import image_transform, image_transform_valid, OurDataLoader

# model
localisation_classes = 4
# optimizer
default_learning_rate = 10e-5
momentum = 0.9  # for SGD
# loss
default_sigma = 15
default_lambdah = 1


def init_model(save_file, learning_rate):
    model = CSLNet(localisation_classes=localisation_classes)
    model.load_state_dict(torch.load(os.path.abspath("weights/resnet50-19c8e357.pth")), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1, patience=50)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': 0,
    }, save_file)


def _load_model(state_dict):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = CSLNet(localisation_classes=localisation_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, device


def train_model(workspace, dataset, segmentation_loss, normalize_heatmap=False, batch_size=2, lambdah=default_lambdah, sigma=default_sigma, non_img_norm_flag=True, learning_rate=default_learning_rate):
    checkpoint = torch.load(os.path.join(workspace, 'csl.pth'))
    model, device = _load_model(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose = True)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    dataset = OurDataLoader(data_dir=dataset, task_type='both', transform=image_transform(p=1),
                            pose_sigma=sigma,
                            normalize_heatmap=normalize_heatmap,
                            seg_type='binary',
                            non_image_norm_flag=non_img_norm_flag)
    epoch = checkpoint['epoch']
    if device == 'cuda':
        del checkpoint
        torch.cuda.empty_cache()
    training = Training(model, dataset, optimizer, scheduler, segmentation_loss, start_epoch=epoch, workspace=workspace,
                        device=device, lambdah=lambdah, batch_size=batch_size)
    training.start()


def call_model(workspace, dataset, normalize_heatmap=False, batch_size=2, sigma=default_sigma, non_img_norm_flag=True):
    checkpoint = torch.load(os.path.join(workspace, 'csl.pth'), map_location=torch.device('cpu'))
    model, device = _load_model(checkpoint['model_state_dict'])
    dataset = OurDataLoader(data_dir=dataset, task_type='both', transform=image_transform_valid(p=1),
                            pose_sigma=sigma,
                            normalize_heatmap=normalize_heatmap,
                            seg_type='binary',
                            non_image_norm_flag=non_img_norm_flag)
    if device == 'cuda':
        del checkpoint
        torch.cuda.empty_cache()

    #model.visualize(dataset, device, batch_size=1)
    model.show_loc_result(dataset, device, batch_size=1)
    #model.show_all_result(dataset, device, batch_size=1)

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--command", "-c", type=str, choices=['init', 'train', 'call'], default='train')
    arg_parser.add_argument("--workspace", "-w", type=str, default='.')
    arg_parser.add_argument("--dataset", "-d", type=str, default='../dataset')
    arg_parser.add_argument("--segloss", "-sl", type=str, choices=['ce', 'dice'], default='ce')
    arg_parser.add_argument("--normalize", "-n", action='store_true', default=False)
    arg_parser.add_argument("--batch", "-b", type=int, default=2)
    arg_parser.add_argument("--lambdah", "-l", type=float, default=default_lambdah)
    arg_parser.add_argument("--sigma", "-s", type=int, default=default_sigma)
    arg_parser.add_argument("--non_img_norm_flag", "-in", action='store_false', default=True)
    arg_parser.add_argument("--learningrate", "-lr", type=float, default=default_learning_rate)

    args = arg_parser.parse_args()

    if args.command == 'init':
        model_out = os.path.join(args.workspace, 'csl.pth')
        init_model(model_out, args.learningrate)
    elif args.command == 'train':
        if args.segloss == 'ce':
            segmentation_loss = Training.LossFunction.SegmentationLoss.cross_entropy
        elif args.segloss == 'dice':
            segmentation_loss = Training.LossFunction.SegmentationLoss.dice
        else:
            raise ValueError
        train_model(args.workspace, args.dataset, segmentation_loss,
                    normalize_heatmap=args.normalize, batch_size=args.batch, lambdah=args.lambdah, sigma=args.sigma, non_img_norm_flag=args.non_img_norm_flag, learning_rate=args.learningrate)
    elif args.command == 'call':
        call_model(args.workspace, args.dataset, normalize_heatmap=args.normalize, batch_size=args.batch, sigma=args.sigma, non_img_norm_flag=args.non_img_norm_flag)

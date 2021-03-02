from argparse import ArgumentParser
from copy import deepcopy

import torch
from pytorch_nndct.apis import torch_quantizer

from csl.csl_run import default_sigma
from csl.data_loader import OurDataLoader, image_transform_valid, train_val_dataset
from csl.net import CSLNet
from util.evaluate import get_threshold_score


def evaluate(model, dataloader, use_cuda=False):
    if use_cuda:
        model = model.to("cuda")
    model.eval()
    total = 0
    loss = 0
    segmentation_accuracy = 0
    keypoints_f1 = 0
    for index, batch in enumerate(dataloader):
        inputs, target = batch
        if use_cuda:
            inputs = inputs.to("cuda")
            target = target.to("cuda")
        target = target.permute(0, 3, 1, 2)
        target_segmentation, target_localisation = torch.split(target, [1, 4], dim=1)
        batch_size = inputs.shape[0]
        total += batch_size
        output = model(inputs)
        output_segmentation, output_localisation = output
        upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        output_segmentation = upsample(output_segmentation)
        output_localisation = upsample(output_localisation)
        # loss
        weights = torch.where(target_localisation > 0.0001, torch.full_like(target_localisation, 5),
                              torch.full_like(target_localisation, 1))
        all_mse = (output_localisation - target_localisation) ** 2
        weighted_mse = all_mse * weights
        localisation_loss = weighted_mse.sum() / (4 * batch_size)
        loss += (torch.nn.functional.binary_cross_entropy_with_logits(output_segmentation, target_segmentation) +
                 localisation_loss).item()
        # segmentation accuracy
        mask_incorrect = (output_segmentation >= 0.5) != (target_segmentation >= 0.5)
        segmentation_accuracy += 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        # keypoints_f1
        image_pairs = []
        for pred_single, target_single in zip(output_localisation.split(1, 0), target_localisation.split(1, 0)):
            for pred_class, target_class in zip(pred_single.split(1, 1), target_single.split(1, 1)):
                pred_np = pred_class.squeeze().detach().cpu().numpy()
                target_np = target_class.squeeze().detach().cpu().numpy()
                image_pairs.append((target_np, pred_np))
        threshold_score, true_positive, false_positive, false_negative = get_threshold_score(image_pairs, None)
        epsilon = 10e-6
        precision = float(true_positive) / (true_positive + false_positive + epsilon)
        recall = float(true_positive) / (true_positive + false_negative + epsilon)
        keypoints_f1 += 2 / ((1 / (recall + epsilon)) + (1 / (precision + epsilon)))
        print("\revaluating {}/{}".format(index, len(dataloader)), end="")
    print("")
    return segmentation_accuracy / total, keypoints_f1 / total, loss / total


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--weights", "-w", type=str, required=True)
    arg_parser.add_argument("--batch_size", "-b", type=int, default=1)
    arg_parser.add_argument("--quant_mode", "-q", choices=["calib", "test", "float"], required=True)
    arg_parser.add_argument("--deploy", "-d", action="store_true")
    arg_parser.add_argument("--finetune", "-ft", action="store_true")
    arg_parser.add_argument("--dataset", "-ds", type=str, default=".")
    args = arg_parser.parse_args()
    input = torch.randn((args.batch_size, 3, 512, 960))
    checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
    model = CSLNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = OurDataLoader(data_dir=args.dataset, task_type='both', transform=image_transform_valid(p=1),
                            pose_sigma=default_sigma,
                            normalize_heatmap=True,
                            seg_type='binary',
                            non_image_norm_flag=True)
    loader = train_val_dataset(dataset, validation_split=0, train_batch_size=args.batch_size,
                               valid_batch_size=args.batch_size, shuffle_dataset=True)[0]
    use_cuda = False
    if args.quant_mode != "float":
        quantizer = torch_quantizer(args.quant_mode, model, (input))
        model = quantizer.quant_model
        if args.quant_mode == "calib":
            if args.finetune:
                quantizer.fast_finetune(evaluate, (model, loader))
            quantizer.export_quant_config()

        elif args.quant_mode == "test":
            if args.finetune:
                quantizer.load_ft_param()
            if args.deploy:
                quantizer.export_xmodel(deploy_check=False)
    elif torch.cuda.is_available():
        use_cuda = True

    segmentation_accuracy, keypoint_f1, loss = evaluate(model, loader, use_cuda=use_cuda)
    print("segmentation accuracy: {}\n keypoint f1 score: {}\n loss: {}"
          .format(segmentation_accuracy, keypoint_f1, loss))

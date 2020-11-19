import numpy as np
import torch.nn as nn
import torch


def softmax_my(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def loss_by_np(output, target, weight_np_array):
    loss = 0
    for i in range(512):
        for j in range(960):
            temp_o = output[0, :, i, j]
            temp_t = target[0, :, i, j]
            sft = -np.log(softmax_my(temp_o))

            sft_weighted = np.multiply(sft, weight_np_array)

            loss += sum(np.multiply(temp_t, sft_weighted))
    print(loss)


def loss_by_torch(output, target, weight_np_array):
    output_segmentation = torch.from_numpy(output)

    weight_torch_array = torch.from_numpy(weight_np_array)

    target_argmax = np.array([np.argmax(a, axis=0) for a in target])
    print(target_argmax.shape)
    target_segmentation = torch.tensor(target_argmax)

    segmentation_loss_function = nn.CrossEntropyLoss(weight=weight_torch_array, reduction='sum')
    segmentation_loss = segmentation_loss_function(output_segmentation, target_segmentation)
    print(segmentation_loss)


if __name__ == '__main__':
    weight_array = np.array([1, 3, 3, 0.01])  # weight of each class
    output = np.zeros((1, 4, 512, 960))
    target = np.zeros((1, 4, 512, 960))

    for i in range(512):
        for j in range(960):
            target[0, 2, i, j] = 1  # background

    for i in range(960):
        output[0, 0, 0, i] = 1
        target[0, 0, 1, i] = 1  # set class 1
        # target[0,2,1,i] = 0 #clean corresponding background

    print("==============")

    loss_by_np(output, target, weight_array)
    loss_by_torch(output, target, weight_array)

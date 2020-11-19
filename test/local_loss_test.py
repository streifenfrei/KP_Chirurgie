import numpy as np
import torch.nn as nn
import torch


def loss_by_np(output, target):
    difference = output - target

    mse = np.sum(np.multiply(difference, difference))
    print(mse)


def loss_by_torch(output, target):
    output_tensor = torch.from_numpy(output)
    target_tensor = torch.from_numpy(target)

    localisation_loss_function = nn.MSELoss(reduction='sum')
    localisation_loss = localisation_loss_function(output_tensor, target_tensor)
    print(localisation_loss)


if __name__ == '__main__':
    output = np.random.rand(512, 960, 4) * 10000000 * 512  # 0-1
    target = np.random.rand(512, 960, 4) * 512  # 0-1
    loss_by_np(output, target)  # 8.593053396935769e+24
    loss_by_torch(output, target)

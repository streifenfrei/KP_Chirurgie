
import torch
from torch import nn
import numpy as np


#==================== how to construct indices: ============================#
def construct_indices(after_pooling):
    our_indices = np.zeros_like(after_pooling, dtype = np.int64)
    batch_num, channel_num, row_num, col_num = after_pooling.shape
    for batch_id in range(batch_num):
        for channel_id in range(channel_num):
            for row_id in range(row_num):
                for col_id in range(col_num):
                    our_indices[batch_id, channel_id, row_id, col_id] = col_num * 2 * 2 * row_id + 2 * col_id
    return our_indices
#===========================================================================#


if __name__ == '__main__':

    before_pooling = torch.Tensor([[[[9, 0, 10, 0, 5, 0],
                                     [0, 0, 0, 0,  0, 0],
                                     [6, 0, 4, 0, 11, 0],
                                     [0, 0, 0, 0,  0, 0],
                                     [2, 0, 4, 0, 19, 0],
                                     [0, 0, 0, 0,  0, 0]]]]
                     )
    print(before_pooling)

    pool= nn.MaxPool2d(2,stride=2,return_indices = False)
    after_pooling = pool(before_pooling)
    print('after_pooling:')
    print(after_pooling)


    _our_indices = construct_indices(after_pooling)
    
    
    tensor_for_unpooling = torch.from_numpy(_our_indices)
    print("constructed indices by ourselves:")
    print(tensor_for_unpooling) 

    unpool = nn.MaxUnpool2d(2,stride=2)
    up=unpool(after_pooling,tensor_for_unpooling)
    print("after unpooling")
    print(up)



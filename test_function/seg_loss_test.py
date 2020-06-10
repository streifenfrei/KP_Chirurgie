import numpy as np
import torch.nn as nn
import torch

def softmax_my(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
def loss_by_np(output, target):
    loss = 0
    for i in range(512):
        for j in range(960):
            temp_o = output[0,:,i,j]
            temp_t = target[0,:,i,j]
            sft = -np.log(softmax_my(temp_o))

            loss += sum(np.multiply(temp_t, sft))
    print(loss)
            
def loss_by_torch(output, target):
    output_segmentation = torch.from_numpy(output)
    
    target_argmax = np.array([np.argmax(a, axis = 0) for a in target])

    target_segmentation = torch.tensor(target_argmax)
    
    segmentation_loss_function = nn.CrossEntropyLoss(reduction='sum')
    segmentation_loss = segmentation_loss_function(output_segmentation, target_segmentation)
    print(segmentation_loss)
    
if __name__ == '__main__':
    
    output = np.zeros((1,4,512,960)) 
    target = np.zeros((1,4,512,960)) 
    
    #for i in range(512): 
    #    for j in range(960):
    #        target[0,2,i,j] = 1 #background
            
    for i in range(960):
        output[0,0,0,i] = 1
        target[0,0,1,i] = 1 #set class 1
        #target[0,2,1,i] = 0 #clean corresponding background

    print("==============")

    loss_by_np(output, target)
    loss_by_torch(output, target) #681734.4834

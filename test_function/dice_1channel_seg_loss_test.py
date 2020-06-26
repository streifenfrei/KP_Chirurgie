import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
reference:
https://github.com/rogertrullo/pytorch/blob/e6c0943401335a24c590f64c3b70d21f156ca0e0/torch/nn/functional.py#L708

I modified:
    1. dice_total= -1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz ==> dice_total= 3 - torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
       because we can convert the loss from [-3,0] into [0,3], 
       check this: https://stackoverflow.com/questions/49785133/keras-dice-coefficient-loss-function-is-negative-and-increasing-with-epochs
       and we have 3 channel, so here is 3 instead of 1
       
    2. dice_eso=dice[:,1:] ==> dice_eso=dice[:,0:-1] 
       because our background is the last channel(-1),
       so the dice loss will ignore the background ==> solved the background pixel imbalance issue?

    3. probs=F.softmax(input) ==> probs=F.softmax(input, dim = 1)
       otherwise the code can't compiled, weird.
    
careful: input should not be softmaxed, because in this function there is a probs=F.softmax(input, dim = 1)
       
'''
def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class, last class would be the background 
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"
    smooth = 0.00001
    if(input.shape[1] == 1):
        probs = torch.sigmoid(input)
        print(probs)
    else:    
        probs = F.softmax(input, dim = 1)

    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
        
    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
        
    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c

    dice=2*((num + smooth)/(den1+den2+smooth))
    print(dice.shape)
    if(input.shape[1] == 1):
        dice_eso=dice#we ignore bg dice val, and take the fg
    else:
        dice_eso=dice[:,0:-1]#we ignore bg dice val, and take the fg
    dice_total=1 - torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total

    
def test_loss(output, target):
    output_torch = torch.from_numpy(output)
    target_torch = torch.tensor(target)
    loss = dice_loss(output_torch, target_torch)
    print(loss)
    
    
if __name__ == '__main__':
    
    output = np.ones((3,1,512,960))  # batch, channel, h, w
    target = np.zeros((3,1,512,960)) 
    output *= -10.0
    '''
    for i in range(output.shape[2]): 
        for j in range(output.shape[3]):
            output[0,0,i,j] = -10
    '''
    #output *= -10    
    for i in range(output.shape[3]):
        output[0,0,0,i] = 10.0

        target[0,0,0,i] = 1.0 #set class 0
    
    print("==============")
    test_loss(output, target)


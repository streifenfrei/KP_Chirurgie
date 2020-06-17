import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

from dataLoader import image_transform, OurDataLoader, train_val_dataset

#明天搞
#https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
def non_max_suppression(img_landmark):
    neighborhood_size = 10
    threshold = 0.2
    print(np.amax(img_landmark))
    data = np.array(img_landmark* 256, dtype=int) 

    data_max = filters.maximum_filter(data, neighborhood_size)

    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    xy = []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)//2
        #x.append(x_center)
        y_center = (dy.start + dy.stop - 1)//2    
        #y.append(y_center)
        xy.append([x_center, y_center])
    #print(x,y)
    return np.array(xy)
    #plt.imshow(data)
    #plt.savefig('data.png', bbox_inches = 'tight')

    #plt.autoscale(False)
    #plt.plot(x,y, 'r+', markersize=15)
    #plt.savefig('result.png', bbox_inches = 'tight')
    
# https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian    
def nearest_neighbors(target_xy, array):
    return np.argsort(np.array([np.linalg.norm(target_xy-x) for x in array]))[0]
    
def findNN(image_label, image_predicted):
    xy_label = non_max_suppression(imags_label)
    xy_predict = non_max_suppression(imags_predict)
    
    pair_list = []
    for i in xy_label:
        nn_id = nearest_neighbors(i, xy_predict)
        pair_list.append([i, xy_predict[nn_id]])
    pair_array = np.array(pair_list)
    print(pair_array)
    
    fig=plt.figure(figsize=(12, 6))
    fig.add_subplot(1,3,1)    
    plt.imshow(imags_label)
    fig.add_subplot(1,3,2)    
    plt.imshow(imags_predict)
    fig.add_subplot(1,3,3)
    plt.imshow(imags_predict + imags_label)
    for pair in pair_array:
        plt.plot(pair[0][0],pair[0][1], 'r+', markersize=15)
        plt.plot(pair[1][0],pair[1][1], 'b+', markersize=15)
        plt.plot([pair[0][0],pair[1][0]], [pair[0][1], pair[1][1]])
    plt.show()
def plot_threshold_score():
    pass
    
'''
TODO: test dice coefficient
''' 
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))
    
    
if __name__ == '__main__':
    dataset = OurDataLoader(data_dir=r'dataset', task_type = 'both', transform=image_transform(p=1), pose_sigma = 5, normalize_heatmap = True)
    train_loader, validation_loader = train_val_dataset(dataset, validation_split = 0.3, train_batch_size = 2, valid_batch_size = 2, shuffle_dataset = True)
    
    
    print("train_batchs: " + str(len(train_loader)))
    print("valid_batchs: " + str(len(validation_loader)))
    
    import matplotlib.pyplot as plt
    
    # Usage Example:
    num_epochs = 1
    for epoch in range(num_epochs):
        # Train:

        print("train set:")
        for batch_index, (image, labels) in enumerate(train_loader):
            print('Epoch: ', epoch, '| Batch_index: ', batch_index, '| image: ',image.shape, '| labels: ', labels.shape)
            break
            
        # Valid
        print("valid set")
        for batch_index, (image, labels) in enumerate(validation_loader):
            print('Epoch: ', epoch, '| Batch_index: ', batch_index, '| image: ',image.shape, '| labels: ', labels.shape)
            
            imags_label = labels[0,:,:,5].view(labels[0].shape[0], labels[0].shape[1]).numpy()
            #imags_predict = labels[1,:,:,4].view(labels[0].shape[0], labels[0].shape[1]).numpy()
            #imags_original = image[0].view(image[0].shape[0], image[0].shape[1], image[0].shape[2]).permute(1, 2, 0)
            imags_predict = np.roll(imags_label, 50, axis=0) 
            imags_predict = np.roll(imags_predict, 50, axis=1) 
            findNN(imags_label, imags_predict)
                
            break
            #fig=plt.figure(figsize=(12, 6))
            #fig.add_subplot(2,3,1)
            #plt.imshow(image[0].view(image[0].shape[0], image[0].shape[1], image[0].shape[2]).permute(1, 2, 0))
            '''
            fig.add_subplot(3,4,2)
            plt.imshow(labels[0,:,:,0].view(labels[0].shape[0], labels[0].shape[1]))
            fig.add_subplot(3,4,3)
            plt.imshow(labels[0,:,:,1].view(labels[0].shape[0], labels[0].shape[1]))
            
            fig.add_subplot(3,4,4)
            plt.imshow(labels[0,:,:,2].view(labels[0].shape[0], labels[0].shape[1]))

            fig.add_subplot(3,4,5)
            plt.imshow(labels[0,:,:,3].view(labels[0].shape[0], labels[0].shape[1]))
            '''

            #fig.add_subplot(2,3,2)
            #plt.imshow(labels[0,:,:,4].view(labels[0].shape[0], labels[0].shape[1]))
            #imags = labels[0,:,:,4].view(labels[0].shape[0], labels[0].shape[1])
            
            
            #fig.add_subplot(2,3,3)
            #plt.imshow(labels[0,:,:,5].view(labels[0].shape[0], labels[0].shape[1]))
            #fig.add_subplot(2,3,4)
            #plt.imshow(labels[0,:,:,6].view(labels[0].shape[0], labels[0].shape[1]))
            #fig.add_subplot(2,3,5)
            #plt.imshow(labels[0,:,:,7].view(labels[0].shape[0], labels[0].shape[1]))
            #plt.show()
            
            
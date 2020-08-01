import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import cv2
from dataLoader import image_transform, OurDataLoader, train_val_dataset, landmark_id_to_name_
    

#https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
def non_max_suppression(img_landmark):
    neighborhood_size = 7
    threshold = 0.4
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

    
# https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian    
def nearest_neighbors(target_xy, array):
    return np.argsort(np.array([np.linalg.norm(target_xy-x) for x in array]))[0]
    
def findNN(image_label, image_predicted, save_name):
    TP = 0
    FP = 0
    FN = 0

    xy_label = non_max_suppression(image_label)
    image_predicted = applyThreshold(image_predicted, 0.4)
    xy_predict = non_max_suppression(image_predicted)

    pair_list = []
    for i in xy_label:
        if(len(xy_predict)==0):
            FN += 1
            continue
        nn_id = nearest_neighbors(i, xy_predict)

        if np.linalg.norm(i-xy_predict[nn_id])>50:
            continue

        pair_list.append([i, xy_predict[nn_id]])
        xy_predict = np.delete(xy_predict, nn_id, 0)
        TP += 1

    if(len(xy_predict)!=0):
        FP = len(xy_predict)

    pair_array = np.array(pair_list)
    print(pair_array)

    if save_name == None:
        return pair_array,[TP,FP,FN]

    fig=plt.figure(figsize=(12, 6))
    fig.add_subplot(1,3,1)    
    plt.imshow(image_label)
    fig.add_subplot(1,3,2)    
    plt.imshow(image_predicted)
    fig.add_subplot(1,3,3)
    plt.imshow(image_predicted + image_label)
    for pair in pair_array:
        plt.plot(pair[0][0],pair[0][1], 'r+', markersize=15)
        plt.plot(pair[1][0],pair[1][1], 'b+', markersize=15)
        plt.plot([pair[0][0],pair[1][0]], [pair[0][1], pair[1][1]])
    plt.savefig(save_name)
    plt.close('all')
    return pair_array,[TP,FP,FN]
    
    
# TODO: test    
# https://note.nkmk.me/en/python-opencv-numpy-alpha-blend-mask/   

 
def plotOverlayImages(ori_image, seg_image, loc_images, save_name):
    ori_image = np.float32(ori_image) * 255
    seg_image = np.float32(seg_image) * 255
    seg_image_3_channel = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)
    lower =(80, 80, 80) # lower bound for each channel
    upper = (255, 255, 255) # upper bound for each channel

    # create the mask and use it to change the colors
    mask = cv2.inRange(seg_image_3_channel, lower, upper)
    seg_image_3_channel[mask != 0] = [50, 50, 0]
    #print('seg_image_3_channel: ',seg_image_3_channel.shape)
    seg_overlap = cv2.addWeighted(ori_image, 0.7, seg_image_3_channel, 0.3, 70)
    
    cv2.imwrite(r"../out/segmented_weighted.jpg", seg_overlap) # for seg
    # https://www.cnblogs.com/darkknightzh/p/6117528.html color reference
    loc_class_list = ['firebrick', 'midnightblue', 'sandybrown', 'linen']
    label_class_list = ['lightsteelblue', 'royalblue', 'blue', 'midnightblue']
    loc_classes = len(loc_images)
    img = plt.imread(r"../out/segmented_weighted.jpg")
    #img = cv2.cvtColor(seg_overlap, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(15,15))
    ax.imshow(img)
    for loc_class_ in range(loc_classes):
        image_predicted = loc_images[loc_class_]
        image_predicted = applyThreshold(image_predicted, 0.4)
        xy_predict = non_max_suppression(np.float32(image_predicted))  
        for xy in xy_predict:
            ax.plot(xy[0],xy[1], color = loc_class_list[loc_class_], marker = '.', markersize=7, label=landmark_id_to_name_[loc_class_ + 1])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="landmark type")
    #ax.legend()
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close('all')   
    
    
def applyThreshold(image_predicted, thres_value):
    image_predicted_thresholded = (image_predicted > thres_value) * image_predicted
    return image_predicted_thresholded
    
    
def plot_threshold_score(all_image_pairs,threshold_list = [10, 20, 30, 40, 50]):
    threshold_count_dict = {}
    all_TP = 0
    all_FP = 0
    all_FN = 0
    for threshold in threshold_list:
        threshold_count_dict[threshold] = 0
    
    for (image_label, image_predicted) in all_image_pairs:
          
        pair_array, static_list = findNN(image_label, image_predicted, None)
        all_TP += static_list[0]
        all_FP += static_list[1]
        all_FN += static_list[2]
        for pair in pair_array:
            for threshold in threshold_list:
                if (np.sqrt(  (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2  ) < threshold):
                    threshold_count_dict[threshold] += 1
    
    x = threshold_list
    y = [threshold_count_dict[i]/(all_TP + 0.0) for i in threshold_count_dict.keys()]
    print('TP, FP, FN:')
    print(all_TP, all_FP, all_FN)
    print(y)
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10,5))
    plt.title("distance threshold score")
    plt.xlabel("threshold")
    plt.ylabel("counts")
    plt.plot(x, y,'b-',label="CSL model")
    
    plt.plot(x,y,'b^-')
    plt.legend()
    plt.grid(True)
    plt.savefig('../out/score_threshold.png')
    
    
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
    train_loader, validation_loader = train_val_dataset(dataset, validation_split = 0.0, train_batch_size = 3, valid_batch_size = 2, shuffle_dataset = True)
    
    
    print("train_batchs: " + str(len(train_loader)))
    print("valid_batchs: " + str(len(validation_loader)))
    
    import matplotlib.pyplot as plt
    
    # Usage Example:
    num_epochs = 1
    test_list = []
    for epoch in range(num_epochs):
        # Train:
        
        print("train set:")
        for batch_index, (image, labels) in enumerate(train_loader):
            print('Epoch: ', epoch, '| Batch_index: ', batch_index, '| image: ',image.shape, '| labels: ', labels.shape)

            imags_label = labels[0,:,:,4].view(labels[0].shape[0], labels[0].shape[1]).numpy()
            imags_predict = np.roll(imags_label, 20, axis=0) 
            imags_predict = np.roll(imags_predict, 10, axis=1) 
            test_list.append((imags_label, imags_predict))
            
            
            imags_label_1 = labels[2,:,:,4].view(labels[1].shape[0], labels[1].shape[1]).numpy()
            imags_predict_1 = np.roll(imags_label_1, 10, axis=0) 
            imags_predict_1 = np.roll(imags_predict_1, 0, axis=1) 
            test_list.append((imags_label_1, imags_predict_1))
            '''
            fig=plt.figure(figsize=(12, 6))
            fig.add_subplot(6,4,1)
            plt.imshow(image[0].view(image[0].shape[0], image[0].shape[1], image[0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(6,4,2)
            plt.imshow(labels[0,:,:,4].view(labels[0].shape[0], labels[0].shape[1]))
            
            fig.add_subplot(6,4,3)
            plt.imshow(labels[0,:,:,5].view(labels[0].shape[0], labels[0].shape[1]))
            
            fig.add_subplot(6,4,4)
            plt.imshow(labels[0,:,:,6].view(labels[0].shape[0], labels[0].shape[1]))

            fig.add_subplot(6,4,5)
            plt.imshow(labels[0,:,:,7].view(labels[0].shape[0], labels[0].shape[1]))
            
            
            fig.add_subplot(6,4,6)
            plt.imshow(image[1].view(image[0].shape[0], image[0].shape[1], image[0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(6,4,7)
            plt.imshow(labels[1,:,:,4].view(labels[0].shape[0], labels[0].shape[1]))
            
            fig.add_subplot(6,4,8)
            plt.imshow(labels[1,:,:,5].view(labels[0].shape[0], labels[0].shape[1]))
            
            fig.add_subplot(6,4,9)
            plt.imshow(labels[1,:,:,6].view(labels[0].shape[0], labels[0].shape[1]))

            fig.add_subplot(6,4,10)
            plt.imshow(labels[1,:,:,7].view(labels[0].shape[0], labels[0].shape[1]))
            plt.show()
            '''
            #test_list.append((labels[0,:,:,4].numpy(), 1))
            #findNN(imags_label, imags_predict)

        plot_threshold_score(test_list)
            
        # Valid
        print("valid set")
        for batch_index, (image, labels) in enumerate(validation_loader):
            print('Epoch: ', epoch, '| Batch_index: ', batch_index, '| image: ',image.shape, '| labels: ', labels.shape)
            

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
            
            
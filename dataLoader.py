import torch
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
import glob
import json
from scipy.ndimage.filters import gaussian_filter

# for image augmentation
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
    
)

# for transforming base64 to image array
import base64
import io
import PIL.Image
import PIL.ImageDraw

# generate unicode
import uuid

from torch.utils.data import DataLoader

class_name_to_id_ = {
'grasper':1,
'scissors':2,
'needle_holder ':3
}

landmark_name_to_id_ = {
'jaw':1,
'center':2,
'joint':3,
'shaft':4
}

class OurDataLoader(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', task_type='both', class_name_to_id = class_name_to_id_, landmark_name_to_id = landmark_name_to_id_, pose_sigma = 7): 
        '''
        constructor of OurDataLoader
        @ param:
            1. data_dir: r'.\\dataset' (the dir of dataset)
            2. transform: image_transform (for augmentation of image)
            3. mode: 'train' or 'test' (return label or not)
            4. task_type: 'pose' or 'segmentation' or 'both' (return which type of label)
        '''
        self.all_json_name_list = find_all_json(data_dir)  # json file name list
        self.transform = transform
        self.mode = mode
        self.task_type = task_type
        self.class_name_to_id = class_name_to_id
        self.landmark_name_to_id = landmark_name_to_id
        self.pose_sigma = pose_sigma

    def __len__(self):
        '''
        return size of dataset
        '''
        return len(self.all_json_name_list)

    def __getitem__(self, idx):
        '''
        get next image (w/o label)
        '''
        data_name = self.all_json_name_list[idx]

        image, shapes = load_image(data_name)
        if self.task_type == 'segmentation':  # not binary but 4 channels: 4 instruments
            mask = load_mask(image.shape, shapes, self.class_name_to_id)

            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]
            if self.mode == 'train':
                return img_to_tensor(image), torch.from_numpy(mask).long()
            else:
                return img_to_tensor(image), str(img_file_name)
                
                
        elif self.task_type == 'pose':
            mask = load_pose(image.shape, shapes, self.landmark_name_to_id, self.pose_sigma)
            data = {"image": image, "mask": mask}
            # TODO: test if works or not with pose information
            augmented = self.transform(**data)
            image, pose = augmented["image"], augmented["mask"]

            if self.mode == 'train':
                return img_to_tensor(image), torch.from_numpy(pose).float()
            else:
                return img_to_tensor(image), str(img_file_name)
                
        # TODO: how to deal with this part?
        elif self.task_type == 'both':
            mask = load_both(image.shape, shapes, self.class_name_to_id, self.landmark_name_to_id, self.pose_sigma)
            print(image.shape)
            print(mask.shape)
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            
            image, both_labels = augmented["image"], augmented["mask"]
            #both_labels = mask_transform(both_labels)

            if self.mode == 'train':
                return img_to_tensor(image), torch.from_numpy(both_labels).float()
            else:
                return img_to_tensor(image), str(img_file_name)
            


def find_all_json(json_dir):
    '''
    find all the data in data dir
    '''
    all_json_names = glob.glob(json_dir+'/*.json')
    return all_json_names


def image_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=100, min_width=100, p=0.5),
        RandomCrop(height=512, width=960, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        #RandomAffine(30)
    ], p=p)
    

def img_b64_to_arr(img_b64):
    '''
    convert base64 to image array
    reference: https://github.com/wkentaro/labelme/blob/master/labelme/cli/json_to_dataset.py
    '''
    img_data = base64.b64decode(img_b64)
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


# TODO: test how it works    
def load_image(json_file):
    data = json.load(open(json_file))
    imageData = data.get('imageData')
    img = img_b64_to_arr(imageData)
    shapes = data['shapes']
    return img, shapes


# ==================================#
def polygons_to_mask(img_shape, points, shape_type=None,
                     line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    assert len(xy) > 2, 'Polygon must have points more than 2'
    draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask    


def point_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
    cx, cy = xy[0]
    #r = point_size
    draw.point([cx , cy], fill=1)   #only draw 1 pixel
    mask = np.array(mask, dtype=bool)
    return mask  
 
        
def shapes_to_label(img_shape, shapes, label_name_to_value, task_type):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        group_id = shape.get('group_id')
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get('shape_type', None)
        if shape_type == 'polygon' and task_type == 'segmentation':
            cls_name = label
            instance = (cls_name, group_id)

            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]

            mask = polygons_to_mask(img_shape[:2], points, shape_type)
            cls[mask] = cls_id
            ins[mask] = ins_id
        elif shape_type == 'point' and task_type == 'pose':
            cls_name = label
            instance = (cls_name, group_id)

            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]

            mask = point_to_mask(img_shape[:2], points, shape_type)
            cls[mask] = cls_id
            ins[mask] = ins_id
    return cls, ins
#==================================#
# load the mask   
def load_mask(img_shape, shapes, class_name_to_id):
    cls, ins = shapes_to_label(
                img_shape=img_shape,
                shapes=shapes,
                label_name_to_value=class_name_to_id,
                task_type = 'segmentation'
            )
    seg_image = np.zeros_like(ins) 
    
    seg_image[cls == 1] = 1 #first channel: grapser
    seg_image = np.reshape(seg_image,(seg_image.shape[0], seg_image.shape[1], 1))
    for each_class in class_name_to_id:
        if each_class != 'grasper':
            seg_image_cls = np.zeros_like(ins)
            seg_image_cls[cls == class_name_to_id[each_class]] = 1
            seg_image_cls = np.reshape(seg_image_cls,(1, seg_image_cls.shape[0], seg_image_cls.shape[1]))
            seg_image = np.dstack((seg_image,seg_image_cls))
    return seg_image

def max_gaussian_help(cls, pose_sigma, landmark_id):    
    non_zero_coords =  np.transpose(np.where(cls == landmark_id))

    seg_image = np.zeros_like(cls, dtype = float)
    seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1],1))
    seg_image_temp = np.zeros_like(seg_image, dtype = float)
    for every_point in non_zero_coords:
        seg_image_zeros = np.zeros_like(seg_image_temp, dtype = float)
        seg_image_zeros[every_point[0], every_point[1],0] = 1.0 
        seg_image_zeros = gaussian_filter(seg_image_zeros, sigma = pose_sigma)
        #print(seg_image_zeros.shape)
        seg_image = np.dstack((seg_image,seg_image_zeros))
    #print(seg_image.shape)
    #print("=============")    
    seg_image = np.max(seg_image, axis = 2)
    
    seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1], 1))

    return seg_image
    
def load_pose(img_shape, shapes, landmark_name_to_id, pose_sigma):
    cls, ins = shapes_to_label(
                img_shape=img_shape,
                shapes=shapes,
                label_name_to_value=landmark_name_to_id,
                task_type = 'pose'
            )
    seg_image = max_gaussian_help(cls, pose_sigma, 1)
    #seg_image[cls == 1] = 1 #first channel: jaw
    #non_zero_coords = np.transpose(np.where(cls == 1))
    #seg_image_temp = seg_image[:].reshape(seg_image.shape[0], seg_image.shape[1]),1)
    #seg_image_zeros = np.zeros_like(ins, dtype = float)
    #for every_point in non_zero_coords:
    #    seg_image_zeros[every_point[0], every_point[1]] = 1.0 
    #    seg_image_zeros = gaussian_filter(seg_image_zeros, sigma = pose_sigma)
    #    seg_image_temp = np.dstack((seg_image_temp,seg_image_zeros))
    
    #seg_image = np.max(seg_image_temp, axis = 2)
    
    #seg_image = gaussian_filter(seg_image, sigma = pose_sigma)
    
    #if seg_image.max() > 0:
    #    seg_image *= (1.0/seg_image.max())
        
    #seg_image = np.reshape(seg_image,(seg_image.shape[0], seg_image.shape[1], 1))
    for each_class in landmark_name_to_id:
        if each_class != 'jaw':
            seg_image_cls = np.zeros_like(ins, dtype = float)
            seg_image_cls = max_gaussian_help(cls, pose_sigma, landmark_name_to_id[each_class])
            #seg_image_cls[cls == landmark_name_to_id[each_class]] = 1
            #apply gaussian filter:
            #seg_image_cls = gaussian_filter(seg_image_cls, sigma = pose_sigma)
            #normalization? ==> not now
            #if seg_image_cls.max() > 0:
            #    seg_image_cls *= (1.0/seg_image_cls.max())
            #seg_image_cls = np.reshape(seg_image_cls,(seg_image_cls.shape[0], seg_image_cls.shape[1]),1)
            seg_image = np.dstack((seg_image,seg_image_cls))
    return seg_image
    
def load_both(img_shape, shapes, class_name_to_id, landmark_name_to_id, pose_sigma):

    cls_seg, ins_seg = shapes_to_label(
                img_shape=img_shape,
                shapes=shapes,
                label_name_to_value=class_name_to_id,
                task_type = 'segmentation'
            )
    cls_pose, ins_pose = shapes_to_label(
                img_shape=img_shape,
                shapes=shapes,
                label_name_to_value=landmark_name_to_id,
                task_type = 'pose'
            )
    seg_image = np.zeros_like(ins_seg, dtype = float)
    
    seg_image[cls_seg == 1] = 1 #first channel: grapser
    seg_image = np.reshape(seg_image,(seg_image.shape[0], seg_image.shape[1], 1))
    for each_class in class_name_to_id:
        if each_class != 'grasper':
            seg_image_cls = np.zeros_like(ins_seg)
            seg_image_cls[cls_seg == class_name_to_id[each_class]] = 1
            seg_image_cls = np.reshape(seg_image_cls,(seg_image_cls.shape[0], seg_image_cls.shape[1], 1))
            seg_image = np.dstack((seg_image,seg_image_cls))

    for each_class in landmark_name_to_id:
        seg_image_cls = np.zeros_like(ins_pose, dtype = float)
        seg_image_cls[cls_pose == landmark_name_to_id[each_class]] = 1.0
        #gaussian
        seg_image_cls = gaussian_filter(seg_image_cls, sigma = pose_sigma)

        seg_image_cls = np.reshape(seg_image_cls,(seg_image_cls.shape[0], seg_image_cls.shape[1], 1))
        seg_image = np.dstack((seg_image,seg_image_cls))

    return seg_image 



    
if __name__ == '__main__':
    test1 = DataLoader(
            dataset=OurDataLoader(data_dir=r'dataset', task_type = 'pose', transform=image_transform(p=1)),
            shuffle=True,
            batch_size=2,
            pin_memory=torch.cuda.is_available()
        )

    for epoch in range(2):
        for step, (batchX, batchY) in enumerate(test1):
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batchX.shape, '| batch y: ', batchY.shape)
    
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12, 6))
    for step, (batchX, batchY) in enumerate(test1):
        '''
        fig.add_subplot(2,4,1)
        plt.imshow(batchX[0].view(batchX[0].shape[0], batchX[0].shape[1], batchX[0].shape[2]).permute(1, 2, 0))

        fig.add_subplot(2,4,2)
        plt.imshow(batchY[0,:,:,0].view(batchY[0].shape[0], batchY[0].shape[1]))
        fig.add_subplot(2,4,3)
        plt.imshow(batchY[0,:,:,1].view(batchY[0].shape[0], batchY[0].shape[1]))
        
        
        fig.add_subplot(2,4,4)
        plt.imshow(batchY[0,:,:,2].view(batchY[0].shape[0], batchY[0].shape[1]))

        fig.add_subplot(2,4,5)
        plt.imshow(batchY[0,:,:,3].view(batchY[0].shape[0], batchY[0].shape[1]))
        non_zero_coords =  np.transpose(np.nonzero(batchY[0,:,:,3].view(batchY[0].shape[0], batchY[0].shape[1]).numpy()))
        print(non_zero_coords)
        for xy in non_zero_coords:
            print(xy[0],xy[1])
            print(batchY[0,:,:,3].numpy()[xy[0],xy[1]])

        fig.add_subplot(2,4,6)
        plt.imshow(batchY[0,:,:,4].view(batchY[0].shape[0], batchY[0].shape[1]))
        fig.add_subplot(2,4,7)
        plt.imshow(batchY[0,:,:,5].view(batchY[0].shape[0], batchY[0].shape[1]))
        fig.add_subplot(2,4,8)
        plt.imshow(batchY[0,:,:,6].view(batchY[0].shape[0], batchY[0].shape[1]))
        '''
        fig.add_subplot(2,3,1)
        plt.imshow(batchX[0].view(batchX[0].shape[0], batchX[0].shape[1], batchX[0].shape[2]).permute(1, 2, 0))

        fig.add_subplot(2,3,2)
        plt.imshow(batchY[0,:,:,0].view(batchY[0].shape[0], batchY[0].shape[1]))
        non_zero_coords =  np.transpose(np.nonzero(batchY[0,:,:,0].view(batchY[0].shape[0], batchY[0].shape[1]).numpy()))
        #print(non_zero_coords)
        #for xy in non_zero_coords:
        #    print(xy[0],xy[1])
        #    print(batchY[0,:,:,0].numpy()[xy[0],xy[1]])
        #print("====================")
        print(np.max(batchY[0,:,:,0].view(batchY[0].shape[0], batchY[0].shape[1]).numpy()))
        fig.add_subplot(2,3,3)
        plt.imshow(batchY[0,:,:,1].view(batchY[0].shape[0], batchY[0].shape[1]))
        
        
        fig.add_subplot(2,3,4)
        plt.imshow(batchY[0,:,:,2].view(batchY[0].shape[0], batchY[0].shape[1]))        
        
        plt.show()
        break

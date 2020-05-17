import torch
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
import glob
import json



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
'__ignore__':-1,
'_background_':0,
'grasper':1,
'scissors':2,
'needle_holder ':3
}


class OurDataLoader(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', task_type='segmentation', class_name_to_id = class_name_to_id_): 
        '''
        constructor of OurDataLoader
        @ param:
            1. data_dir: r'.\\dataset' (the dir of dataset)
            2. transform: image_transform (for augmentation of image)
            3. mode: 'train' or 'test' (return label or not)
            4. task_type: 'pose' or 'segmentation' or 'both' (return which type of label)
        '''
        self.all_json_name_list = find_all_json(data_dir) #json file name list
        self.transform = transform
        self.mode = mode
        self.task_type = task_type
        self.class_name_to_id = class_name_to_id

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
        
        image,shapes = load_image(data_name)
        if self.task_type == 'segmentation': # not binary but 4 channels: 4 instruments
            label = load_mask(image.shape, shapes, self.class_name_to_id)
            
            data = {"image": image, "label": label}
            #augmented = self.transform(**data)
            augmented = data
            image, mask = augmented["image"], augmented["label"]
            if self.mode == 'train':
                return img_to_tensor(image), torch.from_numpy(mask).long()
            else:
                # TODO: why dose it looks like this?
                return img_to_tensor(image), str(img_file_name)
                
                
        elif self.task_type == 'pose':
            label = load_pose(data_name)
            data = {"image": image, "label": label}
            # TODO: test if works or not with pose information
            #augmented = self.transform(**data)
            augmented = data
            image, pose = augmented["image"], augmented["label"]
            
            # TODO: how to deal with pose label?
            if self.mode == 'train':
                pass
            else:
                pass
                
        # TODO: how to deal with this part?
        elif self.task_type == 'both':
            pass




def find_all_json(json_dir):
    '''
    find all the data in data dir
    '''
    all_json_names = glob.glob(json_dir+'\\*.json')
    return all_json_names

    
def image_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=100, min_width=100, p=1),
        RandomCrop(height=100, width=100, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(p=1)
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


#==================================#
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
    
def shapes_to_label(img_shape, shapes, label_name_to_value):
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
        if shape_type == 'polygon':
            cls_name = label
            instance = (cls_name, group_id)

            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]

            mask = polygons_to_mask(img_shape[:2], points, shape_type)
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
            )
    ins[cls == -1] = 0  # ignore it.
    return ins

    
if __name__ == '__main__':
    test1 = DataLoader(
            dataset=OurDataLoader(data_dir=r'.\\dataset', transform=image_transform(p=1)),
            shuffle=True,
            batch_size=2,
            pin_memory=torch.cuda.is_available()
        )

    for epoch in range(3):
        for step, (batchX, batchY) in enumerate(test1):
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batchX.shape, '| batch y: ', batchY.shape)
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(8, 8))
    for step, (batchX, batchY) in enumerate(test1):
        fig.add_subplot(1,2,1)
        plt.imshow(batchX.view(batchX[0].shape[1], batchX[0].shape[2], batchX[0].shape[3]).permute(1, 2, 0))
        fig.add_subplot(1,2,2)
        plt.imshow(batchY.view(batchY[0].shape[1], batchY[0].shape[2]).permute(0, 1))
        plt.show()
        break
    
    
    
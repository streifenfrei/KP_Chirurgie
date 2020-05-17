import torch
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor, ToTensorV2
import glob
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches,  lines

import random
import itertools
import colorsys

# for transforming base64 to image array
import base64
import io
import PIL.Image
import PIL.ImageDraw

# generate unicode
import uuid

from torch.utils.data import DataLoader

# possible classes that occur in our dataset
class_name_to_id_ = {
'__ignore__':-1,
'_background_':0,
'grasper':1,
'scissors':2,
'needle_holder ':3
}


class MaskrcnnDataLoader(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', task_type='segmentation',
                 class_name_to_id=class_name_to_id_):
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

    def __len__(self):
        '''
        return size of dataset
        '''
        return len(self.all_json_name_list)

    def __getitem__(self, idx):
        '''
        get next image (w/o label)
        the implementation is basically taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        '''
        data_name = self.all_json_name_list[idx]

        image, shapes = load_image(data_name)
        if self.task_type == 'segmentation':  # not binary but 4 channels: 4 instruments
            label = load_mask(image.shape, shapes, self.class_name_to_id)

            # construction of the boxes (ground truth for region proposal)
            obj_ids = np.unique(label)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            num_objs = len(obj_ids)
            masks = label == obj_ids[:, None, None]
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            #converting to thensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.from_numpy(label).long()
            labels = torch.ones((num_objs,), dtype=torch.int64)

            # creating the dict srtucture for the maskrcnn groundtruth
            # see https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn
            target={}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            #addded code



            if self.mode == 'train':
                return img_to_tensor(image), target
            else:
                # TODO: why dose it looks like this?
                img_file_name = os.path.basename(data_name)
                return img_to_tensor(image), str(img_file_name)


        elif self.task_type == 'pose':
            label = load_pose(data_name)
            data = {"image": image, "label": label}
            # TODO: test if works or not with pose information
            # augmented = self.transform(**data)
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
    all_json_names = glob.glob(json_dir+'*.json')
    return all_json_names


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


# ==================================#

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
        dataset=MaskrcnnDataLoader(data_dir=r'./dataset/', ),
        shuffle=False,
        batch_size=1,
        pin_memory=torch.cuda.is_available()
    )


    fig = plt.figure(figsize=(8, 8))
    for step, (batchX, batchY) in enumerate(test1):
        fig.add_subplot(1, 2, 1)
        # show the original image
        plt.imshow(batchX.view(batchX.shape[1], batchX.shape[2], batchX.shape[3]).permute(1, 2, 0))

        # Get the current reference
        ax = plt.gca()
        boxes = batchY['boxes']
        for i in boxes: #TODO found a prettier accessing technique
            for index, box in enumerate(i):
                x1, y1, x2, y2 = box.numpy()
                # drawing the rectangle to show the ground truth for region prorposal
                plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7,
                                              linestyle="dashed",edgecolor='r', facecolor='none'))
        fig.add_subplot(1, 2, 2)
        # show the mask
        plt.imshow(batchY['masks'].view(batchY['masks'].shape[1], batchY['masks'].shape[2]).permute(0, 1))

        plt.show()
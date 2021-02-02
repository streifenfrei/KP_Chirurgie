import torch
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
import glob
import json
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms_t

# for image augmentation
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    RandomCrop,
    CenterCrop,
    RandomBrightnessContrast,
    Blur,
    HueSaturationValue,
    ShiftScaleRotate
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
    'grasper': 1,
    'scissors': 2,
    'needle_holder': 3,
    'needleholder': 3,
    'background': 0
}

landmark_name_to_id_ = {
    'jaw': 1,
    'center': 2,
    'joint': 3,
    'shaft': 4
}


class OurDataLoader(Dataset):
    def __init__(self, data_dir, transform=None, task_type='both', class_name_to_id=None,
                 landmark_name_to_id=None, pose_sigma=7, normalize_heatmap=True, seg_type='binary',
                 non_image_norm_flag=True):
        """
        constructor of OurDataLoader
        @ param:
            1. data_dir: r'.\\dataset' (the dir of dataset)
            2. transform: image_transform (for augmentation of image)
            3. mode: 'train' or 'test' (return label or not)
            4. task_type: 'pose' or 'segmentation' or 'both' (return which type of label)
        """
        if landmark_name_to_id is None:
            landmark_name_to_id = landmark_name_to_id_
        if class_name_to_id is None:
            class_name_to_id = class_name_to_id_
        self.all_json_name_list = find_all_json(data_dir)  # json file name list
        self.transform = transform
        self.task_type = task_type
        self.class_name_to_id = class_name_to_id
        self.landmark_name_to_id = landmark_name_to_id
        self.pose_sigma = pose_sigma
        self.normalize_heatmap = normalize_heatmap
        self.seg_type = seg_type
        self.non_image_norm_flag = non_image_norm_flag

    def __len__(self):
        """
        return size of dataset
        """
        return len(self.all_json_name_list)

    def __getitem__(self, idx):
        """
        get next image (w/o label)
        """
        data_name = self.all_json_name_list[idx]

        img, shapes = load_image(data_name)
        if self.task_type == 'segmentation':  # not binary but 4 channels: 4 instruments
            mask = load_mask(img.shape, shapes, self.class_name_to_id, self.seg_type)

            data = {"image": img, "mask": mask}
            augmented = self.transform(**data)
            img, mask = augmented["image"], augmented["mask"]
            return img_to_tensor(img), torch.from_numpy(mask).long()

        elif self.task_type == 'pose':
            mask = load_pose(img.shape, shapes, self.landmark_name_to_id, self.pose_sigma, self.normalize_heatmap)
            data = {"image": img, "mask": mask}
            # TODO: test if works or not with pose information
            augmented = self.transform(**data)
            img, pose = augmented["image"], augmented["mask"]

            return img_to_tensor(img), torch.from_numpy(pose).float()

        # TODO: how to deal with this part?
        elif self.task_type == 'both':
            mask = load_both(img.shape, shapes, self.class_name_to_id, self.landmark_name_to_id, self.pose_sigma,
                             self.normalize_heatmap, self.seg_type)
            data = {"image": img, "mask": mask}
            augmented = self.transform(**data)

            img, both_labels = augmented["image"], augmented["mask"]

            if self.non_image_norm_flag:
                print("I'm normalize the image!")
                tf = image_norm()
                img = tf(img)
                return img, torch.from_numpy(both_labels).float()
            else:
                return img_to_tensor(img), torch.from_numpy(both_labels).float()


def find_all_json(json_dir):
    """
    find all the data in data dir
    """
    all_json_names = glob.glob(json_dir + '/**/*.json', recursive=True)
    return all_json_names


def image_transform(p=1):
    return Compose([
        # PadIfNeeded(min_height=100, min_width=100, p=0.5),
        RandomCrop(height=512, width=960, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        Blur(p=0.5),
        HueSaturationValue(p=0.5),
        ShiftScaleRotate(scale_limit=0.1, rotate_limit=30, border_mode=0)
    ], p=p)


def image_transform_valid(p=1):
    return Compose([
        CenterCrop(height=512, width=960, p=1),
    ], p=p)


def image_norm():
    return transforms_t.Compose([
        transforms_t.ToTensor(),
        transforms_t.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def img_b64_to_arr(img_b64):
    """
    convert base64 to image array
    reference: https://github.com/wkentaro/labelme/blob/master/labelme/cli/json_to_dataset.py
    """
    img_data = base64.b64decode(img_b64)
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


def load_image(json_file):
    data = json.load(open(json_file))
    image_data = data.get('imageData')

    if image_data is None:
        img_name = json_file[0:-4] + 'png'
        with open(img_name, "rb") as f:
            b = io.BytesIO(f.read())
            img_pil = PIL.Image.open(b)
            img = np.array(img_pil)
    else:
        img = img_b64_to_arr(image_data)

    shapes = data['shapes']
    return img, shapes


def polygons_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    assert len(xy) > 2, 'Polygon must have points more than 2'
    draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def point_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
    cx, cy = xy[0]
    # r = point_size
    draw.point([cx, cy], fill=1)  # only draw 1 pixel
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

            mask = polygons_to_mask(img_shape[:2], points)
            cls[mask] = cls_id
            ins[mask] = ins_id
        elif shape_type == 'point' and task_type == 'pose':
            cls_name = label
            instance = (cls_name, group_id)

            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]

            mask = point_to_mask(img_shape[:2], points)
            cls[mask] = cls_id
            ins[mask] = ins_id
    return cls, ins


# load the mask
def load_mask(img_shape, shapes, class_name_to_id, seg_type):
    cls, ins = shapes_to_label(
        img_shape=img_shape,
        shapes=shapes,
        label_name_to_value=class_name_to_id,
        task_type='segmentation'
    )
    seg_image = np.zeros_like(ins)

    if seg_type == 'binary':
        seg_image[cls == class_name_to_id['grasper']] = 1
        seg_image[cls == class_name_to_id['scissors']] = 1
        seg_image[cls == class_name_to_id['needle_holder']] = 1
        seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1], 1))

    else:
        seg_image[cls == 1] = 1  # first channel: grapser
        seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1], 1))
        for each_class in class_name_to_id:
            if each_class != 'grasper':
                seg_image_cls = np.zeros_like(ins)
                seg_image_cls[cls == class_name_to_id[each_class]] = 1
                seg_image_cls = np.reshape(seg_image_cls, (1, seg_image_cls.shape[0], seg_image_cls.shape[1]))
                seg_image = np.dstack((seg_image, seg_image_cls))
    return seg_image


def max_gaussian_help(cls, pose_sigma, landmark_id):
    non_zero_coords = np.transpose(np.where(cls == landmark_id))

    seg_image = np.zeros_like(cls, dtype=float)
    seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1], 1))

    seg_image_temp = np.zeros_like(seg_image, dtype=float)
    for every_point in non_zero_coords:
        seg_image_zeros = np.zeros_like(seg_image_temp, dtype=float)
        seg_image_zeros[every_point[0], every_point[1], 0] = 1.0
        seg_image_zeros = gaussian_filter(seg_image_zeros, sigma=pose_sigma)
        seg_image = np.dstack((seg_image, seg_image_zeros))

    seg_image = np.max(seg_image, axis=2)

    seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1], 1))

    return seg_image


def load_pose(img_shape, shapes, landmark_name_to_id, pose_sigma, normalize_heatmap):
    cls, ins = shapes_to_label(
        img_shape=img_shape,
        shapes=shapes,
        label_name_to_value=landmark_name_to_id,
        task_type='pose'
    )
    seg_image = max_gaussian_help(cls, pose_sigma, 1)
    for each_class in landmark_name_to_id:
        if each_class != 'jaw':
            seg_image_cls = max_gaussian_help(cls, pose_sigma, landmark_name_to_id[each_class])
            seg_image = np.dstack((seg_image, seg_image_cls))
    if normalize_heatmap:
        seg_image = seg_image * 2 * np.pi * pose_sigma * pose_sigma
    else:
        seg_image_cls = seg_image_cls * np.sqrt(2 * np.pi) * pose_sigma

    return seg_image


def load_both(img_shape, shapes, class_name_to_id, landmark_name_to_id, pose_sigma, normalize_heatmap, seg_type):
    cls_seg, ins_seg = shapes_to_label(
        img_shape=img_shape,
        shapes=shapes,
        label_name_to_value=class_name_to_id,
        task_type='segmentation'
    )
    cls_pose, ins_pose = shapes_to_label(
        img_shape=img_shape,
        shapes=shapes,
        label_name_to_value=landmark_name_to_id,
        task_type='pose'
    )
    seg_image = np.zeros_like(ins_seg, dtype=float)

    if seg_type == 'binary':
        seg_image[cls_seg == class_name_to_id['grasper']] = 1
        seg_image[cls_seg == class_name_to_id['scissors']] = 1
        seg_image[cls_seg == class_name_to_id['needle_holder']] = 1
        seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1], 1))

    else:
        seg_image[cls_seg == 1] = 1  # first channel: grapser
        seg_image = np.reshape(seg_image, (seg_image.shape[0], seg_image.shape[1], 1))
        for each_class in class_name_to_id:
            if each_class != 'grasper':
                seg_image_cls = np.zeros_like(ins_seg)
                seg_image_cls[cls_seg == class_name_to_id[each_class]] = 1
                seg_image_cls = np.reshape(seg_image_cls, (seg_image_cls.shape[0], seg_image_cls.shape[1], 1))
                seg_image = np.dstack((seg_image, seg_image_cls))

    for each_class in landmark_name_to_id:
        seg_image_cls = max_gaussian_help(cls_pose, pose_sigma, landmark_name_to_id[each_class])
        if normalize_heatmap == True:
            seg_image_cls = seg_image_cls * 2 * np.pi * pose_sigma * pose_sigma
            # am = np.amax(heatmap)
            # seg_image_cls /= am
        else:
            seg_image_cls = seg_image_cls * np.sqrt(2 * np.pi) * pose_sigma
        # print(seg_image_cls.shape)
        seg_image = np.dstack((seg_image, seg_image_cls))
    return seg_image


def train_val_dataset(dataset_list, validation_split=0.2, train_batch_size=16, valid_batch_size=16,
                      shuffle_dataset=True):
    random_seed = 42
    # Creating data indices for training and validation splits:

    if isinstance(dataset_list, list):
        dataset_size = 0
        for dataset in dataset_list:
            dataset_size += len(dataset)

        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(ConcatDataset(dataset_list), batch_size=train_batch_size,
                                                   sampler=train_sampler, num_workers=8, pin_memory=True)
        validation_loader = torch.utils.data.DataLoader(ConcatDataset(dataset_list), batch_size=valid_batch_size,
                                                        sampler=valid_sampler, num_workers=8, pin_memory=True)
    else:
        dataset = dataset_list
        random_seed = 42
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)

        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size,
                                                   sampler=train_sampler, num_workers=8, pin_memory=True)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=valid_batch_size,
                                                        sampler=valid_sampler, num_workers=8, pin_memory=True)
    return train_loader, validation_loader


def prepare_batch(batch, segmentation_classes, localisation_classes):
    inputs, target = batch
    target = target.permute(0, 3, 1, 2)
    target_segmentation, target_localisation = torch.split(target, [segmentation_classes, localisation_classes], dim=1)
    target_segmentation_np = np.array([np.argmax(a, axis=0) for a in target_segmentation.numpy()])
    target_segmentation = torch.tensor(target_segmentation_np)
    return inputs, (target_segmentation, target_localisation)


if __name__ == '__main__':

    dataset1 = OurDataLoader(data_dir=r'../data/Data1', task_type='both', transform=image_transform(p=1), pose_sigma=15,
                             normalize_heatmap=True, seg_type='binary')
    dataset2 = OurDataLoader(data_dir=r'../data/Data2', task_type='both', transform=image_transform(p=1), pose_sigma=15,
                             normalize_heatmap=True, seg_type='binary')

    # load 1 dataset, or as list: both are ok 
    train_loader, validation_loader = train_val_dataset([dataset1, dataset2], validation_split=0.3, train_batch_size=2,
                                                        valid_batch_size=2, shuffle_dataset=True)

    print("train_batchs: " + str(len(train_loader)))
    print("valid_batchs: " + str(len(validation_loader)))

    import matplotlib.pyplot as plt

    # Usage Example:
    num_epochs = 10
    for epoch in range(num_epochs):
        # Train:
        print("train set:")
        for batch_index, (image, labels) in enumerate(train_loader):
            print('Epoch: ', epoch, '| Batch_index: ', batch_index, '| image: ', image.shape, '| labels: ',
                  labels.shape)

            fig = plt.figure(figsize=(12, 6))
            fig.add_subplot(3, 3, 1)
            plt.imshow(image[0].view(image[0].shape[0], image[0].shape[1], image[0].shape[2]).permute(1, 2, 0))
            for row in image[0, 1, :, :].numpy():
                print(max(row))
            fig.add_subplot(3, 3, 2)
            plt.imshow(labels[0, :, :, 0].view(labels[0].shape[0], labels[0].shape[1]))
            fig.add_subplot(3, 3, 3)
            plt.imshow(labels[0, :, :, 1].view(labels[0].shape[0], labels[0].shape[1]))
            # print(labels.shape)
            # for row in labels[0,:,:,1].view(labels[0].shape[0], labels[0].shape[1]).numpy():
            #    print(max(row))

            fig.add_subplot(3, 3, 4)
            plt.imshow(labels[0, :, :, 2].view(labels[0].shape[0], labels[0].shape[1]))

            fig.add_subplot(3, 3, 5)
            plt.imshow(labels[0, :, :, 3].view(labels[0].shape[0], labels[0].shape[1]))
            '''
            fig.add_subplot(3,3,6)
            plt.imshow(labels[0,:,:,4].view(labels[0].shape[0], labels[0].shape[1]))
            fig.add_subplot(3,3,7)
            plt.imshow(labels[0,:,:,5].view(labels[0].shape[0], labels[0].shape[1]))
            '''
            plt.show()
            break

            '''
            fig.add_subplot(3,4,8)
            plt.imshow(labels[0,:,:,6].view(labels[0].shape[0], labels[0].shape[1]))
            fig.add_subplot(3,4,9)
            plt.imshow(labels[0,:,:,7].view(labels[0].shape[0], labels[0].shape[1]))

            '''

        # Valid
        print("valid set")
        for batch_index, (image, labels) in enumerate(validation_loader):
            print('Epoch: ', epoch, '| Batch_index: ', batch_index, '| image: ', image.shape, '| labels: ',
                  labels.shape)

        # break

    '''
    for epoch in range(2):
        for step, (batchX, batchY) in enumerate(test1):
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batchX.shape, '| batch y: ', batchY.shape)
    
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12, 6))
    for step, (batchX, batchY) in enumerate(test1):
        
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
   
        fig.add_subplot(2,4,6)
        plt.imshow(batchY[0,:,:,4].view(batchY[0].shape[0], batchY[0].shape[1]))
        fig.add_subplot(2,4,7)
        plt.imshow(batchY[0,:,:,5].view(batchY[0].shape[0], batchY[0].shape[1]))
        fig.add_subplot(2,4,8)
        plt.imshow(batchY[0,:,:,6].view(batchY[0].shape[0], batchY[0].shape[1]))
    '''
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
    '''

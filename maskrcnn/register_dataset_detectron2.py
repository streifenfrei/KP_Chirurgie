import os
import json
import numpy as np
from detectron2.structures import BoxMode
import dataLoader as dl
import glob
from pprint import pprint
from typing import List, Dict
from PIL import Image

# Detectron 2 requires an integer instead of a class label as a string
# This is a dict for mapping the class labels to integers
mapping = {
        'scissors': 0,
        'needle_holder': 1,
        'needleholder':1,
        'grasper':2
    }


def save_img_from_base(filename: str, img_data: np.ndarray, path_to_save: str) -> None:
    """
    Saving images from numpy ndarray
    This images will be further used for training of Detectron2 model

    Args:
        img_data: contains image as numpy array that will be converted and saved further

    Returns:
        nothing to return, image will be just saved
    """
    image = Image.fromarray(img_data)
    image.save(f"{path_to_save}" + filename)


def create_desription_single_file(json_file: str, for_json: dict, path_to_save: str, save_image: bool = False) -> dict:
    """
    Creating a description for a single file according to the example of
    https://github.com/matterport/Mask_RCNN/releases/

    Args:
        json_file: file from whcih we get the whole infromation is an originally generated annotated
        by labelme file
        for_json: Dict that will be filled with information
        path_to_save: where to save image and the dict dumped further as json
        save_image: mode for image saving from base64 (is for the first time used)

    Returns:
        A filled dict `for_json` wil a filled information about the json_file
        {'frame_00000.png':
        {'fileref': '',
         'size': 1555200,
         'height': 540,
         'width': 960,
         'base64_img_data': '',
         'file_attributes': {},
         'regions':
         {'8': {'name': 'polygon',
                'all_points_x': [0.0, 435.037037037037, 441.2098765432099, 452.32098765432096, ,.....
            354.7901234567901, 125.0, 1.0864197530864197],
                'all_points_y': [521.0, 297.1358024691358, 284.1728395061728, 285.4074074074074,.......
            333.55555555555554, 343.4320987654321, 357.6296296296296, 415.037037037037, 539.0, 538.4938271604938],
             '  region_attributes': {},
                'label': 'grasper'},

    """
    data = json.load(open(json_file))
    try:
        imageData = data.get('imageData')
        if (imageData is not None) and (imageData != ''):
            img = dl.img_b64_to_arr(imageData)
            height, width = img.shape[:2]
            shapes = data['shapes']
            filename = data['imagePath']
            filename_without_extension = os.path.splitext(os.path.basename(json_file))[0]
            single_image = {}
            filename = filename_without_extension + '.png'
            for_json[str(filename)] = single_image
            single_image['fileref'] = ""
            single_image['filename']= filename
            single_image['size'] = img.size
            single_image['height'] = height
            single_image['width'] = width
            single_image['base64_img_data'] = ""
            single_image['file_attributes'] = {}
            regions = {}
            single_image['regions'] = regions
            if save_image:
                save_img_from_base(filename, img_data=img, path_to_save=path_to_save)

            for index, shape in enumerate(shapes):
                if shape['shape_type'] == 'polygon':
                    shape_attr = dict()
                    shape_attr['name'] = 'polygon'
                    all_points_x = list()
                    all_points_y = list()
                    for x, y in shape['points']:
                        all_points_x.append(x)
                        all_points_y.append(y)

                    shape_attr['all_points_x'] = all_points_x
                    shape_attr['all_points_y'] = all_points_y
                    print(shape['label'])
                    shape_attr['label'] = mapping.get(shape['label'])
                    shape_attribute = dict()
                    shape_attribute['shape_attributes'] = shape_attr
                    shape_attribute["region_attributes"] = {}
                    regions[str(index)] = shape_attribute
    except ValueError:
        print(f'Failed to process {json_file =}')
    return for_json


def create_desription_json_for_detectron_registration(json_folder: List[str],
                                                      path_to_save: str, save_image=False) -> Dict:
    """
    Creates a description for the objects that are in the directory according to
    the Google Colab from the official Docu of Detectron2
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5# -> Prepare the Dataset.

    Here was made a structure like in baloon dataset description file `baloon/train/via_region_data.json`
    https://github.com/matterport/Mask_RCNN/releases/
    """
    for_json = dict()

    for index, json_file in enumerate(json_folder):
        for_json = create_desription_single_file(json_file, for_json, path_to_save=path_to_save, save_image=save_image)
        # print(f'Finished {index=} for {json_file=}')

    with open(f'{path_to_save}/dataset_registration_detectron2.json', 'w') as f:
        json.dump(for_json, f)
    return for_json


# json_img  = glob.glob('../dataset/*.json')
json_img = sorted(glob.glob('/Users/chernykh_alexander/Yandex.Disk.localized/CloudTUD/Komp_CHRIRURGIE/instruments/train_json/*.json'))

json_back = create_desription_json_for_detectron_registration(json_img,
                                                              path_to_save='/Users/chernykh_alexander/Yandex.Disk.localized/CloudTUD/Komp_CHRIRURGIE/instruments/train/',
                                                              save_image=False)
pprint(json_back)

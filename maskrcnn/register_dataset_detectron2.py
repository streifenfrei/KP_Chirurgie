import os
import numpy as np
import json
from detectron2.structures import BoxMode
import dataLoader as dl
import cv2
import glob
from pprint import pprint
#
# '{"34020010494_e5cb88e1c4_k.jpg1115004":' \
# '   {"fileref":"",' \
# '   "size":1115004,' \
# '   "filename":"34020010494_e5cb88e1c4_k.jpg","base64_img_data":"","file_attributes":{},' \
# 'regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":[1020,1000,994,1003,1023,1050,1089,1134,1190,1265,1321,1361,1403,1428,1442,1445,1441,1427,1400,1361,1316,1269,1228,1198,1207,1210,1190,1177,1172,1174,1170,1153,1127,1104,1061,1032,1020],' \
# '"all_points_y":[963,899,841,787,738,700,663,638,621,619,643,672,720,765,800,860,896,942,990,1035,1079,1112,1129,1134,1144,1153,1166,1166,1150,1136,1129,1122,1112,1084,1037,989,963]},\
# "region_attributes":{}}}}'


def create_desription_json_for_detectron_registration(json_folder):
    for_json = dict()
    # img, shapes = dl.load_image(json)
    for json_file in json_folder:
        data = json.load(open(json_file))
        imageData = data.get('imageData')
        img = dl.img_b64_to_arr(imageData)
        height, width = img.shape[:2]
        image_size = img.size
        shapes = data['shapes']
        filename = data['imagePath']


        single_image = {}
        for_json[str(filename)] = single_image
        single_image['fileref'] = ""
        single_image['size'] = image_size
        single_image['height'] = height
        single_image['width'] = width
        single_image['base64_img_data'] = ""
        single_image['file_attributes'] = {}
        regions = {}
        single_image['regions'] = regions
        file_attributes = {}
        name='polygon'


        for index, shape in enumerate(shapes):
            if shape['shape_type']=='polygon':
                shape_attributes= dict()
                shape_attributes['name']= 'polygon'
                all_points_x = list()
                all_points_y = list()
                for x, y  in shape['points']:
                    all_points_x.append(x)
                    all_points_y.append(y)

                shape_attributes['all_points_x'] = all_points_x
                shape_attributes['all_points_y'] = all_points_y
                shape_attributes["region_attributes"] = {}
                shape_attributes['label'] = shape['label']
                regions[str(index)] = shape_attributes
                # print(f'{shape_attributes=}')
                # print(shape['label'])
    return for_json



json_img  = glob.glob('../dataset/*.json')
json_back = create_desription_json_for_detectron_registration(json_img)
pprint(json_back)
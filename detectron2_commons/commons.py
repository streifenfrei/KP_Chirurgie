import json
import os
from typing import List

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import detectron2


def get_instrument_dicts(img_dir: str,
                      json_with_desription_name: str = "dataset_registration_detectron2.json") -> List[dict]:
    json_file = os.path.join(img_dir, json_with_desription_name)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        record["height"] = v['height']
        record["width"] = v['width']
        record["file_name"] = filename
        record["image_id"] = idx

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": anno['category_id'],
                "keypoints_csl": anno['keypoints_csl']
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset_and_metadata(path_to_data: str,
                                  classes_list: List[str]) -> detectron2.data.catalog.Metadata:
    for d in ["train", "val"]:
        DatasetCatalog.register("instruments_" + d, lambda d=d: get_instrument_dicts(path_to_data + d))
        MetadataCatalog.get("instruments_" + d).set(thing_classes=classes_list)
    instruments_metadata = MetadataCatalog.get("instruments_train")
    return instruments_metadata

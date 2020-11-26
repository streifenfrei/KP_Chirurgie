import contextlib
import io
import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO


class Evaluator(DatasetEvaluator):

    def __init__(self, dataset_name):
        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                prediction["instances"] = output["instances"].to("cpu")
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to("cpu")
            self._predictions.append(prediction)

    def evaluate(self):
        self._eval_segmentation()
        return {}

    def _eval_segmentation(self):
        for prediction_dict in self._predictions:
            ann_ids = self._coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
            anno = self._coco_api.loadAnns(ann_ids)
            print(anno)
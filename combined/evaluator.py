from detectron2.evaluation import DatasetEvaluator


class Evaluator(DatasetEvaluator):

    def __init__(self):
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            print(output["instances"].gt_keypoints)
            if "instances" in output:
                prediction["instances"] = output["instances"].to("cpu")
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to("cpu")
            self._predictions.append(prediction)

    def evaluate(self):
        return {}
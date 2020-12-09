from detectron2.evaluation import COCOEvaluator
import torch
import numpy as np

class Evaluator(COCOEvaluator):

    def __init__(self, dataset_name, distributed=True, tasks=("bbox", "segm", "keypoints"), **kwargs):
        super().__init__(dataset_name, distributed=distributed, tasks=tasks, use_fast_impl=False, **kwargs)

    def process(self, inputs, outputs):
        instances = outputs[0]["instances"]
        all_keypoints = []
        for keypoints_per_instance in instances.pred_keypoints:
            new_keys = []
            for keypoints_per_class in keypoints_per_instance:
                keypoints_per_class.sort(key=lambda x: np.linalg.norm(x))
                for i in range(2):
                    keypoint = keypoints_per_class[i] if i < len(keypoints_per_class) else None
                    if keypoint is None:
                        new_keys.append(torch.Tensor([0.0, 0.0, 0.1]))
                    else:
                        new_keys.append(torch.Tensor([keypoint[0], keypoint[1], 1.0]))
            all_keypoints.append(torch.stack(new_keys))
        instances.pred_keypoints = torch.stack(all_keypoints)
        super().process(inputs, outputs)

    def evaluate(self, img_ids=None):
        return super().evaluate(img_ids=img_ids)


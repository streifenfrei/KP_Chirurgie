from detectron2.utils.visualizer import Visualizer, GenericMask
import numpy as np

# for visualising the local result, by huxi
from networkx.drawing.tests.test_pylab import mpl

from evaluate import applyThreshold, non_max_suppression


class CSLVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        super().draw_instance_predictions(predictions)
        loc_class_list = ['firebrick', 'midnightblue', 'sandybrown', 'linen']
        print(predictions.pred_loc)
        for keypoints_per_instance in predictions.pred_loc:
            for i, keypoints_per_class in enumerate(keypoints_per_instance):
                for xy in keypoints_per_class:
                    self.output.ax.add_patch(mpl.patches.Circle(xy, radius=5, fill=True, color=loc_class_list[i]))
        return self.output


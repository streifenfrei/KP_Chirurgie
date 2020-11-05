from detectron2.utils.visualizer import Visualizer, GenericMask
import numpy as np

# for visualising the local result, by huxi
from networkx.drawing.tests.test_pylab import mpl

from evaluate import applyThreshold, non_max_suppression


class CSLVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        super().draw_instance_predictions(predictions)
        # === huxi's code: for visualisation of csl keypoint === #
        loc_class_list = ['firebrick', 'midnightblue', 'sandybrown', 'linen']
        my_heatmap_tensor = predictions.pred_loc.cpu().detach().numpy()
        import torch
        print(torch.max(predictions.pred_loc))
        if my_heatmap_tensor is not None:
            for i in range(my_heatmap_tensor.shape[0]):
                for j in range(my_heatmap_tensor.shape[1]):
                    heatmap = my_heatmap_tensor[i, j, :, :]
                    heatmap_thres = applyThreshold(heatmap, 0.8)
                    xy_predict = non_max_suppression(np.float32(heatmap_thres))
                    for xy in xy_predict:
                        self.output.ax.add_patch(mpl.patches.Circle(xy, radius=5, fill=True, color=loc_class_list[j]))

        # === huxi's code: for visualisation of csl keypoint end === #
        return self.output


from detectron2.utils.visualizer import Visualizer
from networkx.drawing.tests.test_pylab import mpl


class CSLVisualizer(Visualizer):
    """
    Custom visualizer for drawing nice keypoints onto the image
    """
    def draw_instance_predictions(self, predictions):
        super().draw_instance_predictions(predictions)
        loc_class_list = ['firebrick', 'midnightblue', 'sandybrown', 'linen']
        for keypoints_per_instance in predictions.pred_loc:
            for i, keypoints_per_class in enumerate(keypoints_per_instance):
                for xy in keypoints_per_class:
                    self.output.ax.add_patch(mpl.patches.Circle(xy, radius=5, fill=True, color=loc_class_list[i]))
        return self.output


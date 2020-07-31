from detectron2.utils.visualizer import Visualizer


class CSLVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        super().draw_instance_predictions(predictions)
        # TODO add keypoints to visualisation
        # -> predictions is an Instances object containing the heatmaps in the shape (N, 4, H, W)
        # (N = found instances, H = height of image, W = width of image)
        # it is found in the .pred_loc field -> predictions.pred_loc
        # -> self.output is a VisImage object containing the input image and the overlays of the masks etc...
        # there the heatmaps / keypoints have to be added somehow
        # (the super method might be helpful too see how the other stuff was added)
        return self.output
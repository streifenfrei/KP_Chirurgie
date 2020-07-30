from detectron2.utils.visualizer import Visualizer


class CSLVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        super().draw_instance_predictions(predictions)
        return self.output
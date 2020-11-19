import os
import cv2
from argparse import ArgumentParser
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode

from combined.combined_run import load_config
from combined.visualizer import CSLVisualizer

"""
Processes a video and visualizes the predictions of the combined network.
"""


def process_video(video, cfg, metadata):
    capture = cv2.VideoCapture(video)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    root, ext = os.path.splitext(video)
    out_name = "{0}_pred{1}".format(root, ext)
    writer = cv2.VideoWriter(
        out_name,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        capture.get(cv2.CAP_PROP_FPS),
        (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    current_frame = 0
    predictor = DefaultPredictor(cfg)
    try:
        while capture.grab():
            success, frame = capture.retrieve()
            if success:
                out = predictor(frame)
                v = CSLVisualizer(frame[:, :, ::-1],
                                  metadata=metadata,
                                  instance_mode=ColorMode.SEGMENTATION
                                  )
                out = v.draw_instance_predictions(out["instances"].to("cpu"))
                writer.write(out.get_image()[:, :, ::-1])
                print("\rframe {0} / {1}".format(current_frame, frame_count), end='')
            current_frame += 1
    finally:
        capture.release()
        writer.release()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, default='configs/default.yaml')
    arg_parser.add_argument("--video", "-v", type=str, required=True)
    args = arg_parser.parse_args()
    cfg = load_config(config_path=args.config)
    metadata = MetadataCatalog.get("temp")
    metadata.set(thing_classes=cfg.VISUALIZER.CLASS_NAMES)
    metadata.set(thing_colors=cfg.VISUALIZER.INSTANCE_COLORS)
    metadata.set(keypoint_colors=cfg.VISUALIZER.KEYPOINT_COLORS)
    process_video(args.video, cfg, metadata)

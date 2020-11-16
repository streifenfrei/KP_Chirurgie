from detectron2.config import CfgNode as CN
from combined.modeling.heads.roi_heads import *
from combined.modeling.heads.csl_head import *
from combined.modeling.heads.csl_pooler import *
from combined.modeling.meta_arch import *


# add new csl config parameter keys to the cfg (the values are actually irrelevant as they are overwritten by the .yaml)
def add_csl_config(cfg):
    cfg.MODEL.CSL_HEAD = CN()
    cfg.MODEL.CSL_HEAD.NAME = "CSLHead"
    cfg.MODEL.CSL_HEAD.IN_FEATURES = ["p3", "p4", "p5", "p6"]
    cfg.MODEL.CSL_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.CSL_HEAD.LOCALISATION_CLASSES = 4
    cfg.MODEL.CSL_HEAD.HM_PREPROCESSING = "align"
    cfg.MODEL.CSL_HEAD.LOC_WEIGHT = 5.
    cfg.MODEL.CSL_HEAD.LOSS_TYPE = "weighted_mse"
    cfg.MODEL.CSL_HEAD.LAMBDA = 1.
    cfg.MODEL.CSL_HEAD.SIGMA = 7
    cfg.MODEL.CSL_HEAD.EVALUATION = CN()
    cfg.MODEL.CSL_HEAD.EVALUATION.EPSILON = 1e-6
    cfg.MODEL.CSL_HEAD.EVALUATION.THRESHOLD_SCORE_LIST = [10, 20, 30, 40, 50]
    cfg.MODEL.POSTPROCESSING = CN()
    cfg.MODEL.POSTPROCESSING.HM_THRESHOLD = 0.8
    cfg.MODEL.POSTPROCESSING.KEYPOINT_LIMITS = [2, 1, 1, 1]
    cfg.VISUALIZER = CN()
    cfg.VISUALIZER.CLASS_NAMES = ['scissors', 'needle_holder', 'grasper']
    cfg.VISUALIZER.INSTANCE_COLORS = [[0, 0, 255], [255, 255, 255], [255, 0, 0]]
    cfg.VISUALIZER.KEYPOINT_COLORS = ['firebrick', 'midnightblue', 'sandybrown', 'linen']

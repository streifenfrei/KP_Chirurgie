from detectron2.config import CfgNode as CN
from combined.heads.roi_heads import *
from combined.heads.csl_head import *
from combined.heads.csl_pooler import *
from combined.meta_arch.combined_arch import *


def add_csl_config(cfg):
    cfg.MODEL.CSL_HEAD = CN()
    cfg.MODEL.CSL_HEAD.NAME = "CSLHead"
    cfg.MODEL.CSL_HEAD.IN_FEATURES = ["p3", "p4", "p5", "p6"]
    cfg.MODEL.CSL_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.CSL_HEAD.LOCALISATION_CLASSES = 4
    cfg.MODEL.CSL_HEAD.LAMBDA = 1.
    cfg.MODEL.CSL_HEAD.SIGMA = 7


from detectron2.config import CfgNode as CN

def add_csl_config(cfg):
    cfg.MODEL.CSL_HEAD = CN()
    cfg.MODEL.CSL_HEAD.NAME = "CSLHead"
    cfg.MODEL.CSL_HEAD.LOCALISATION_CLASSES = 4

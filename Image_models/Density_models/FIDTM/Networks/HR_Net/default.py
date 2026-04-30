from yacs.config import CfgNode as CN

_C = CN()
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args)
    cfg.freeze()
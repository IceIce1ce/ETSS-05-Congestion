from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.SEED = 1
__C.NET = 'HR_Net'
__C.PRE_HR_WEIGHTS = 'pretrained/hrnetv2_w48_imagenet_pretrained.pth'
from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.NET = 'VGG16_FPN' # ['HR_Net', 'VGG16_FPN']
__C.PRE_HR_WEIGHTS = 'pretrained/hrnetv2_w48_imagenet_pretrained.pth'
__C.RESUME = False
__C.RESUME_PATH = ''
__C.GPU_ID = '0'
__C.OPT = 'Adam'
if __C.OPT == 'Adam':
    __C.LR_BASE_NET = 1e-5
    __C.LR_BM_NET = 1e-6
__C.LR_DECAY = 0.99
__C.NUM_EPOCH_LR_DECAY = 4
__C.LR_DECAY_START = 10
__C.MAX_EPOCH = 600
__C.PRINT_FREQ = 20
__C.VAL_DENSE_START = 20
__C.VAL_FREQ = 4
__C.VISIBLE_NUM_IMGS = 1
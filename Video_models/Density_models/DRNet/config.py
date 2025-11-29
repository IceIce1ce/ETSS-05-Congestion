from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.NET = 'VGG16_FPN'
__C.GPU_ID = '0' # '0' or '0,1'
__C.sinkhorn_iterations = 100
__C.FEATURE_DIM = 256
__C.ROI_RADIUS = 4.
__C.LR_Base = 5e-5
__C.LR_Thre = 1e-2
__C.LR_DECAY = 0.95
__C.WEIGHT_DECAY = 1e-5
__C.MAX_EPOCH = 20
__C.PRINT_FREQ = 20
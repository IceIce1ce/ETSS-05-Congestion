from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.NET = 'VGG16_FPN'
__C.continuous= False
__C.RESUME = False
__C.RESUME_PATH = ''
__C.sinkhorn_iterations = 100
__C.FEATURE_DIM = 256
__C.ROI_RADIUS = 4.
__C.LR_Base = 5e-5
__C.LR_Thre = 1e-4
__C.LR_DECAY = 0.95
__C.WEIGHT_DECAY = 1e-5
__C.MAX_EPOCH = 500
__C.PRINT_FREQ = 20

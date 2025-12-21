from easydict import EasyDict as edict
import os

__C = edict()
cfg = __C
__C.encoder = "VGG16_FPN" # VGG16_FPN, ResNet_50_FPN or PCPVT
__C.PRE_TRAIN_COUNTER = ''
__C.GPU_ID = '0,1'
os.environ["CUDA_VISIBLE_DEVICES"] = __C.GPU_ID
__C.cross_attn_embed_dim = 256
__C.cross_attn_num_heads = 4
__C.mlp_ratio = 4
__C.cross_attn_depth = 2
__C.FEATURE_DIM = 256
__C.LR_Base = 1e-5
__C.WEIGHT_DECAY = 1e-6
__C.MAX_EPOCH = 100
__C.VAL_INTERVAL = 10
__C.START_VAL = 20
__C.PRINT_FREQ = 20

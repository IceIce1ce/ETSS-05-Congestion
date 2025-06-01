from easydict import EasyDict as edict

__C_SHHA = edict()
cfg_data = __C_SHHA
__C_SHHA.TRAIN_SIZE = (128, 128)
__C_SHHA.DATA_PATH = 'data/ShanghaiTech/part_A_final'
__C_SHHA.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
__C_SHHA.AUGMENT = 1
__C_SHHA.LOG_PARA = 100.
__C_SHHA.LABEL_FACTOR = 1
__C_SHHA.RESUME_MODEL = ''
__C_SHHA.TRAIN_BATCH_SIZE = 1
__C_SHHA.VAL_BATCH_SIZE = 1
__C_SHHA.EXP_PATH = 'saved_sha'
__C_SHHA.SEED = 640
__C_SHHA.MAX_EPOCH = 200
__C_SHHA.PRINT_FREQ = 10
__C_SHHA.EXP_NAME = 'HMoDE'
__C_SHHA.GPU_ID = '0'

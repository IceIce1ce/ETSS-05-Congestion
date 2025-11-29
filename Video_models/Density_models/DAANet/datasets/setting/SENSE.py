from easydict import EasyDict as edict

__C_SENSE = edict()
cfg_data = __C_SENSE
__C_SENSE.TRAIN_SIZE = (768, 1024)
__C_SENSE.DATA_PATH = 'data/Sense/'
__C_SENSE.TRAIN_LST = 'train.txt'
__C_SENSE.VAL_LST = 'val.txt'
__C_SENSE.TEST_LST = 'test.txt'
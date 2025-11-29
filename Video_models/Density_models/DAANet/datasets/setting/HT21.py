from easydict import EasyDict as edict

__C_HT21 = edict()
cfg_data = __C_HT21
__C_HT21.TRAIN_SIZE = (768, 1024)
__C_HT21.DATA_PATH = 'data/HT21/'
__C_HT21.TRAIN_LST = 'train.txt'
__C_HT21.VAL_LST = 'val.txt'
__C_HT21.TEST_LST = 'test.txt'
from easydict import EasyDict as edict

__C_HT21 = edict()
cfg_data = __C_HT21
__C_HT21.TRAIN_SIZE =(768, 1024)
__C_HT21.DATA_PATH = 'data/HT21'
__C_HT21.TRAIN_LST = 'train.txt'
__C_HT21.VAL_LST =  'val.txt'
__C_HT21.MEAN_STD = ([117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.])
__C_HT21.DEN_FACTOR = 200.
__C_HT21.RESUME_MODEL = ''
__C_HT21.TRAIN_BATCH_SIZE = 2
__C_HT21.TRAIN_FRAME_INTERVALS=(50, 200)
__C_HT21.VAL_BATCH_SIZE = 1
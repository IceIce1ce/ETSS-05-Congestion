from easydict import EasyDict as edict
import numpy as np
import yaml
from ast import literal_eval

_C = edict()
cfg = _C
_C.TAG = 'APGCC'
_C.SEED = 1229
_C.GPU_ID = 0
_C.OUTPUT_DIR = './output/temp/'
_C.VIS = False
# model config
_C.MODEL = edict()
_C.MODEL.ENCODER = 'vgg16_bn' # ['vgg16', 'vgg16_bn']
_C.MODEL.ENCODER_kwargs = {"last_pool": False}
_C.MODEL.DECODER = 'basic' # ['basic', 'IFI']
_C.MODEL.DECODER_kwargs = {"num_classes": 2, "inner_planes": 256, "feat_layers":[3,4], "pos_dim": 2, "ultra_pe": False, "learn_pe": False, "unfold": False, "local": False,
						   "no_aspp": True, "require_grad": True, "out_type": 'Normal', "head_layers":[1024,512,256,256]}
_C.MODEL.STRIDE = 8
_C.MODEL.ROW = 2
_C.MODEL.LINE = 2
_C.MODEL.FROZEN_WEIGHTS = None
# loss config
_C.MODEL.POINT_LOSS_COEF = 0.0002
_C.MODEL.EOS_COEF = 0.5
_C.MODEL.LOSS = ['L2']
_C.MODEL.WEIGHT_DICT = {'loss_ce': 1, 'loss_points': 0., 'loss_aux': 0.} 
_C.MODEL.AUX_EN = False
_C.MODEL.AUX_NUMBER = [1, 1]
_C.MODEL.AUX_RANGE = [1, 4]
_C.MODEL.AUX_kwargs = {'pos_coef': 1., 'neg_coef': 1., 'pos_loc': 0., 'neg_loc': 0.}
# resume training config
_C.RESUME = False
_C.RESUME_PATH = ''
# dataset config
_C.DATASETS = edict()
_C.DATASETS.DATASET = 'SHHA'
_C.DATASETS.DATA_ROOT = './dataset_path/'
_C.DATALOADER = edict()
_C.DATALOADER.AUGUMENTATION = ['Normalize', 'Crop', 'Flip']
_C.DATALOADER.CROP_SIZE = 128
_C.DATALOADER.CROP_NUMBER = 4
_C.DATALOADER.UPPER_BOUNDER = -1
_C.DATALOADER.NUM_WORKERS = 8
# training config
_C.SOLVER = edict()
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.EPOCHS = 3500
_C.SOLVER.LR = 1e-4
_C.SOLVER.LR_BACKBONE = 1e-5
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.LR_DROP = 3500
_C.SOLVER.CLIP_MAX_NORM = 0.1
_C.SOLVER.EVAL_FREQ = 5
_C.SOLVER.LOG_FREQ = 1
# matcher config
_C.MATCHER = edict()
_C.MATCHER.SET_COST_CLASS = 1.
_C.MATCHER.SET_COST_POINT = 0.05
# testing config
_C.TEST = edict()
_C.TEST.THRESHOLD = 0.5
_C.TEST.WEIGHT = ""

def cfg_merge_a2b(a, b):
    if type(a) is not edict and type(a) is not dict:
        raise KeyError('a is not a edict.')
    for k, v in a.items():
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif not isinstance(v, dict) and not isinstance(b[k], type(None)):
                raise ValueError(('Type mismatch ({} vs. {}) for config key: {}').format(type(b[k]), type(v), k))
        if type(v) is edict or type(v) is dict:
            try:
                cfg_merge_a2b(a[k], b[k])
            except:
                print(('Error under config key: {}').format(k))
                raise
        else:
            if v == 'None':
                b[k] = None
            else:
                b[k] = v
    return b

def cfg_from_file(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data

def cfg_from_list(args_opts):
	if len(args_opts) == 0:
		return None
	assert len(args_opts)%2 == 0
	for k, v in zip(args_opts[0::2], args_opts[1::2]):
		key_list = k.split('.')
		d = _C
		for subkey in key_list[:-1]:
			assert subkey in d
			d = d[subkey]
		subkey = key_list[-1]
		assert subkey in d
		try:
			value = literal_eval(v)
		except:
			value = v
		assert type(value) == type(d[subkey]), 'type {} does not match original type {}'.format(type(value), type(d[subkey]))
		d[subkey] = value
	return d

def merge_from_file(cfg, filename):
    file_cfg = cfg_from_file(filename)
    cfg = cfg_merge_a2b(file_cfg, cfg)
    return cfg

def merge_from_list(cfg, args):
	args_cfg = cfg_from_list(args)
	if args_cfg == None:
		return cfg
	else:
		cfg = cfg_merge_a2b(args_cfg, cfg)
		return cfg
from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 1
_C.DATA.INPUT_DIR = ''
_C.DATA.TYPE_DATASET = 'shha'
_C.DATA.PIN_MEMORY = False
_C.DATA.NUM_WORKERS = 8
_C.DATA.MOVE_IMG_TO_MEMORY = False
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet_w48' # hrnet_w48 or vgg19
_C.MODEL.RESUME = ''
_C.MODEL.FACTOR = 1
_C.MODEL.LOSS = 'PML'
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 3000
_C.TRAIN.BASE_LR = 3e-5
_C.TRAIN.BACKBONE_LR = _C.TRAIN.BASE_LR / 2
_C.TRAIN.MIN_LR = 2e-5
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.WARMUP_EPOCHS = _C.TRAIN.EPOCHS // 100
_C.TRAIN.WARMUP_LR = _C.TRAIN.BASE_LR / 1000
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'step'
_C.TEST = CN()
_C.TEST.CROP = True
_C.OUTPUT_DIR = ''
_C.MAX_SAVE_FREQ = 20
_C.MIN_SAVE_FREQ = 5
_C.SAVE_FREQ_FACTOR = (_C.MIN_SAVE_FREQ / _C.MAX_SAVE_FREQ) ** (5 / 3)
_C.PRINT_FREQ = 4
_C.SEED = 2024
_C.EVAL_MODE = False

def update_config(config, args):
    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    if args.type_dataset:
        config.DATA.TYPE_DATASET = args.type_dataset
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.input_dir:
        config.DATA.INPUT_DIR = args.input_dir
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.eval:
        config.EVAL_MODE = True
    config.freeze()

def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config
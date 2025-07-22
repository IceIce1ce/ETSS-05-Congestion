import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lib.opt.train_adv import TrainModel
from src.lib.network.cn import vgg16
import yaml
from easydict import EasyDict as edict
import argparse
import warnings
warnings.filterwarnings("ignore")

def initGenFeatFromCfg(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = edict(yaml.safe_load(f))
    dataset = cfg.DATASET
    data_path = cfg[dataset].DATA_PATH
    target_data_path = cfg[dataset].TARGET_DATA_PATH
    tensor_server_path = cfg[dataset].TENSOR_BOARD_PATH
    pre_trained_path = cfg[dataset].PRE_TRAINED_PATH
    batch_size = cfg[dataset].BATCH_SIZE
    lr = float(cfg[dataset].LEARNING_RATE)
    epoch_num = cfg[dataset].EPOCH_NUM
    steps = cfg[dataset].STEPS
    decay_rate = cfg[dataset].DECAY_RATE
    start_epoch = cfg[dataset].START_EPOCH
    snap_shot = cfg[dataset].SNAP_SHOT
    resize = cfg[dataset].RESIZE
    test_size = cfg[dataset].TEST_SIZE
    return dataset, data_path, target_data_path, tensor_server_path, pre_trained_path, batch_size, lr, epoch_num, steps, decay_rate, start_epoch, snap_shot, resize, test_size

def main(args):
    dataset, data_path, target_data_path, tensor_server_path, pre_trained_path, batch_size, lr, epoch_num, steps, decay_rate, start_epoch, snap_shot, resize, test_size = initGenFeatFromCfg(args.config_file)
    tm = TrainModel(data_path=data_path,target_data_path=target_data_path, batchsize=batch_size, lr=lr, epoch=epoch_num, snap_shot=snap_shot, server_root_path=tensor_server_path, start_epoch=start_epoch, steps=steps, decay_rate=decay_rate, branch=vgg16,
                    pre_trained=pre_trained_path,resize=resize, test_size=test_size)
    tm.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--config_file', type=str, default='configs/shanghaitech_adv.yml')
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    main(args)
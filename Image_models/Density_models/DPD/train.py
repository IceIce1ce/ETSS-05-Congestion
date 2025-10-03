import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from importlib import import_module
import argparse
from FixMatch_trainer import Trainer

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SHHA', choices=['SHHA', 'SHHB', 'QNRF', 'NWPU', 'FDST', 'JHU'])
    parser.add_argument('--output_dir', type=str, default='saved_sha_to_shb')
    parser.add_argument('--vis_dir', type=str, default='vis_sha_to_shb')
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--src_dataset', type=str, default='SHHA')
    parser.add_argument('--src_dir', type=str, default='data/SHHA')
    parser.add_argument('--target_dataset', type=str, default='SHHB')
    parser.add_argument('--target_dir', type=str, default='data/SHHB')
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    setup_seed(args.seed)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    cc_trainer = Trainer(datasetting.cfg_data, args)
    cc_trainer.forward()
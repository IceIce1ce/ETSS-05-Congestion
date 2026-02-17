import warnings
warnings.filterwarnings("ignore")
import random
import argparse
import numpy as np
import torch
from utils.my_trainer import MyTrainer

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', default='Mall', type=str)
    parser.add_argument('--scene', default='scene_001', type=str)
    parser.add_argument('--input_dir', type=str, default='data')
    parser.add_argument('--scene_dataset', default='mall_800_1200')
    parser.add_argument('--output_dir', default='saved_mall', type=str)
    parser.add_argument('--seed', type=str, default=42)
    # training config
    parser.add_argument('--uncertain_thre', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=8)
    parser.add_argument('--loss_weight', type=float, default=0.01)
    parser.add_argument('--iterative_num', type=int, default=20)
    parser.add_argument('--start_iter', type=int, default=4)
    parser.add_argument('--train_num', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    setup_seed(args.seed)
    trainer = MyTrainer(args)
    trainer.setup()
    trainer.train()
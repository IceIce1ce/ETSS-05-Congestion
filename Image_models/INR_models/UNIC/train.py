from utils.regression_trainer_unic import RegTrainer
import argparse
import torch
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', default='data/sha', type=str)
    parser.add_argument('--output_dir', default='saved_sha', type=str)
    parser.add_argument('--is_gray', type=bool, default=False)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    parser.add_argument('--use_background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--background_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=64)
    # training config
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', default='')
    parser.add_argument('--max_model_num', type=int, default=1) # max ckpt to save
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    # testing config
    parser.add_argument('--val_epoch', type=int, default=2)
    parser.add_argument('--val_start', type=int, default=1)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    setup_seed(args.seed)
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
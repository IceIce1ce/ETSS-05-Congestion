import argparse
import torch
from train_helper import Trainer
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--dataset_dir', default='data/nwpu', type=str)
    parser.add_argument('--type_dataset', default='nwpu', choices=['nwpu', 'qnrf', 'sha', 'shb'])
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='saved_nwpu')
    # training config
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=3)
    # testing config
    parser.add_argument('--val_epoch', type=int, default=5)
    parser.add_argument('--val_start', type=int, default=50)
    # loss config
    parser.add_argument('--wot', type=float, default=0.1)
    parser.add_argument('--wtv', type=float, default=0.01)
    parser.add_argument('--reg', type=float, default=10.0)
    parser.add_argument('--num_iter_ot', type=int, default=100)
    parser.add_argument('--norm_cood', type=int, default=0)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    if args.type_dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.type_dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.type_dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.type_dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
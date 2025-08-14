import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import np_transforms as NP_T
from CrowdDataset import TestSeq
from model import STGN
from sklearn.metrics import mean_squared_error,mean_absolute_error

def main(args):
    # test loader
    valid_transf = NP_T.ToTensor()
    if args.type_dataset == 'UCSD':
        args.shape = [360, 480]
        args.max_len = 10
        args.channel = 128
    elif args.type_dataset == 'Mall':
        args.shape = [480, 640]
        args.max_len = 4
        args.channel = 128
    elif args.type_dataset == 'FDST':
        args.shape = [360, 640]
        args.max_len = 4
        args.channel = 128
    elif args.type_dataset == 'Venice':
        args.shape = [360, 640]
        args.max_len = 8
        args.channel = 128
    elif args.type_dataset == 'TRANCOS':
        args.shape = [360, 480]
        args.max_len = 4
        args.channel = 128
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    valid_data = TestSeq(train=False, path=args.input_dir, out_shape=args.shape, transform=valid_transf, gamma=args.gamma, max_len=args.max_len, load_all=args.load_all)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # model
    model = STGN(args).cuda()
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best.pth'), map_location='cuda'))
    preds = {}
    predictions = []
    counts = []
    for i, (X, count, seq_len, names) in enumerate(valid_loader):
        X, count = X.cuda(), count.cuda() # [1, 4, 3, 360, 640], [1, 4, 1, 360, 640]
        with torch.no_grad():
            density_pred, count_pred = model(X) # [1, 4, 1, 45, 80], [1, 4]
        count = count.sum(dim=[2, 3, 4]) # [1, 4]
        count_pred = count_pred.data.cpu().numpy()
        count = count.data.cpu().numpy()
        for i, name in enumerate(names):
            dir_name, img_name = name[0].split('&')
            preds[dir_name + '_' + img_name] = count[0, i]
            predictions.append(count_pred[0, i])
            counts.append(count[0, i])
    mae = mean_absolute_error(predictions, counts)
    rmse = np.sqrt(mean_squared_error(predictions, counts))
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae, rmse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='FDST')
    parser.add_argument('--input_dir', type=str, default='datasets/FDST')
    parser.add_argument('--output_dir', default='saved_fdst', type=str)
    # testing config
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--max_len', default=4, type=int)
    parser.add_argument('--shape', default=[360, 480], nargs='+', type=int)
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    # model config
    parser.add_argument('--channel', default=128, type=int)
    parser.add_argument('--block_num', default=4, type=int)
    parser.add_argument('--agg', action='store_true')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)
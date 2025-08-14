import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import np_transforms as NP_T
from CrowdDataset import CrowdSeq
from model import STGN

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    if args.type_dataset == 'UCSD':
        args.shape = [360, 480]
    elif args.type_dataset == 'Mall':
        args.shape = [480, 640]
    elif args.type_dataset == 'FDST':
        args.shape = [360, 640]
    elif args.type_dataset == 'Venice':
        args.shape = [360, 640]
    elif args.type_dataset == 'TRANCOS':
        args.shape = [360, 480]
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    setup_seed(args.seed)
    # train and test loader
    train_transf = NP_T.ToTensor()
    valid_transf = NP_T.ToTensor()
    train_data = CrowdSeq(train=True, path=args.input_dir, out_shape=args.shape, transform=train_transf, gamma=args.gamma, max_len=args.max_len, load_all=args.load_all, adaptive=args.adaptive)
    valid_data = CrowdSeq(train=False, path=args.input_dir, out_shape=args.shape, transform=valid_transf, gamma=args.gamma, max_len=args.max_len, load_all=args.load_all, adaptive=args.adaptive)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # model
    model = STGN(args).cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # train
    for epoch in range(args.epochs):
        model.train()
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        for i, (X, density, count, seq_len) in enumerate(train_loader): # [1, 4, 3, 360, 640], [1, 4, 1, 45, 80], [1, 4, 1, 360, 640], [4]
            X, density, count, seq_len = X.cuda(), density.cuda(), count.cuda(), seq_len.cuda()
            if random.random() < 0.5:
                X = torch.flip(X, [-1]) # [1, 4, 3, 360, 640]
                density = torch.flip(density, [-1]) # [1, 4, 1, 45, 80]
            density_pred, count_pred = model(X) # [1, 4, 1, 45, 80], [1, 4]
            N = torch.sum(seq_len)
            count = count.sum(dim=[2, 3, 4])
            density_loss = torch.sum((density_pred - density)**2) / (2 * N)
            count_loss = torch.sum((count_pred - count)**2) / (2 * N)
            loss = density_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            with torch.no_grad():
                count_err = torch.sum(torch.abs(count_pred - count)) / N
            count_err_hist.append(count_err.item())
        lr_scheduler.step()
        train_loss = sum(loss_hist) / len(loss_hist)
        train_density_loss = sum(density_loss_hist) / len(density_loss_hist)
        train_count_loss = sum(count_loss_hist) / len(count_loss_hist)
        train_count_err = sum(count_err_hist) / len(count_err_hist)
        print('Epoch: [{}/{}], Train loss: {:.4f}, Density loss: {:.4f}, Count loss: {:.4f}, Count error: {:.4f}'.format(epoch + 1, args.epochs, train_loss, train_density_loss, train_count_loss, train_count_err))
        # test
        model.eval()
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        mse_hist = []
        mae_hist = []
        for i, (X, density, count, seq_len) in enumerate(valid_loader):
            X, density, count, seq_len = X.cuda(), density.cuda(), count.cuda(), seq_len.cuda() # [1, 4, 3, 360, 640], [1, 4, 1, 45, 80], [1, 4, 1, 360, 640], [4]
            with torch.no_grad():
                density_pred, count_pred = model(X) # [1, 4, 1, 45, 80], [1, 4]
            N = torch.sum(seq_len)
            count = count.sum(dim=[2, 3, 4])
            count_loss = torch.sum((count_pred - count)**2) / (2 * N)
            density_loss = torch.sum((density_pred - density)**2) / (2 * N)
            loss = density_loss
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            mae = torch.sum(torch.abs(count_pred - count)) / N
            mae_hist.append(mae.item())
            mse = torch.sqrt(torch.sum((count_pred - count)**2) / N)
            mse_hist.append(mse.item())
        valid_loss = sum(loss_hist) / len(loss_hist)
        valid_density_loss = sum(density_loss_hist) / len(density_loss_hist)
        valid_count_loss = sum(count_loss_hist) / len(count_loss_hist)
        valid_mse = sum(mse_hist) / len(mse_hist)
        valid_mae = sum(mae_hist) / len(mae_hist)
        if epoch == 0:
            min_mae = valid_mae
        else:
            if valid_mae <= min_mae:
                min_mae = valid_mae
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
        print('Epoch: [{}/{}], Val loss: {:.4f}, Density loss: {:.4f}, Count loss: {:.4f}, MAE: {:.2f}, MSE: {:.2f}'.format(epoch + 1, args.epochs, valid_loss, valid_density_loss, valid_count_loss, valid_mae, valid_mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='FDST')
    parser.add_argument('--input_dir', type=str, default='datasets/FDST')
    parser.add_argument('--output_dir', default='saved_fdst', type=str)
    parser.add_argument('--shape', default=[360, 480], nargs='+', type=int)
    parser.add_argument('--seed', default=42, type=int)
    # training config
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--max_len', default=4, type=int)
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--num_workers', type=int, default=6)
    # model config
    parser.add_argument('--channel', default=128, type=int)
    parser.add_argument('--block_num', default=4, type=int)
    parser.add_argument('--agg', action='store_true')
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    main(args)
import argparse
import os.path
import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import random
import numpy as np
import networks
from utils.data import IterLoader
from utils.data.preprocessor import Preprocessor
from utils.data.preprocessor_tran import Preprocessor_tran
from utils.data import transforms as T
import datasets
from utils.trainer_vanilla import Trainer
from evaluator import Evaluator
from utils.clustering.domain_split import domain_split
import warnings
warnings.filterwarnings("ignore")

start_epoch = 0

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def get_data(data_dir, source, num_domains=None):
    if source:
        root = os.path.join(data_dir, 'train_data')
        dataset = datasets.create('CrowdCluster', root, num_domains)
    else:
        root = os.path.join(data_dir, 'test_data')
        dataset = datasets.create('Crowd', root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers):
    normalizer = T.standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([T.RandomHorizontallyFlip(), T.RandomCrop((height, width))])
    img_transformer = T.standard_transforms.Compose([T.standard_transforms.ToTensor(), normalizer])
    gt_transformeer = T.standard_transforms.Compose([T.LabelNormalize(1000.)])
    train_set = sorted(dataset.train)
    train_loader = IterLoader(DataLoader(Preprocessor_tran(train_set, root=dataset.root, main_transform=train_transformer, img_transform=img_transformer, gt_transform=gt_transformeer),
                              batch_size=batch_size, num_workers=workers, sampler=None, shuffle=True, pin_memory=False, drop_last=True), length=None)
    return train_loader

def get_test_loader(dataset, batch_size, workers):
    normalizer = T.standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = None
    img_transformer = T.standard_transforms.Compose([T.standard_transforms.ToTensor(), normalizer])
    gt_transformer = T.standard_transforms.Compose([T.LabelNormalize(1000.)])
    testset = dataset
    test_loader = DataLoader(Preprocessor(testset.train, root=dataset.root, main_transform=test_transformer, img_transform=img_transformer, gt_transform=gt_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)
    return test_loader

def create_model(args):
    model = networks.create(args.arch)
    model.cuda()
    optim = None
    if args.resume:
        global best_mae, best_mse, start_epoch
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        best_mae = checkpoint['mae']
        best_mse = checkpoint['mse']
        start_epoch = checkpoint['epoch']
        optim = checkpoint['optim']
    return model, optim

def online_clustering(source, model, num_clustering):
    pseudo_domain_label = domain_split(source, model, cluster_before=source.clusters, nmb_cluster=num_clustering, method='Kmeans', pca_dim=256, whitening=False, L2norm=False, instance_stat=True)
    source.set_cluster(np.array(pseudo_domain_label))

def main(args):
    global best_mae, best_mse, start_epoch
    best_mae = 100000
    best_mse = 100000
    start_epoch = 0
    cudnn.benchmark = True
    # model
    model, optim = create_model(args)
    # train and test loader
    dataset_src = get_data(args.input_dir, True, args.num_clustering)
    dataset_trg = get_data(args.input_dir, False)
    test_loader = get_test_loader(dataset_trg, args.test_batch_size, args.num_workers)
    evaluator = Evaluator(model)
    # optimizeer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optim is not None:
        optimizer.load_state_dict(optim)
    # loss
    criterion = nn.MSELoss(reduction='sum').cuda()
    trainer = Trainer(args, model, criterion)
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        # online clustering
        if epoch % args.cluster_step == 0:
            online_clustering(dataset_src, model, args.num_clustering)
            train_loaders = []
            for src in dataset_src.subdomains:
                train_loader = get_train_loader(src, args.height, args.width, args.train_batch_size, args.num_workers)
                train_loaders.append(train_loader)
        # train
        trainer.train(epoch, train_loaders, optimizer, print_freq=args.print_freq, train_iters=args.iters)
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mae, mse = evaluator.evaluate(test_loader)
            is_best = (mae < best_mae)
            saved_model = {'state_dict': model.state_dict(), 'epoch': epoch, 'mae': best_mae, 'mse': best_mse, 'optim': optimizer.state_dict()}
            if is_best:
                best_mae = mae
                best_mse = mse
                saved_model['mae'] = best_mae
                saved_model['mse'] = best_mse
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                torch.save(saved_model, osp.join(args.output_dir, 'best.pth.tar'))
            torch.save(saved_model, osp.join(args.output_dir, 'latest.pth.tar'))
            print('Epoch: [{}/{}], MAE: {:.4f}, MSE: {:.4f}, Best MAE: {:.4f}'.format(epoch, args.epochs, mae, mse, best_mae))
    # test
    checkpoint = torch.load(os.path.join(args.output, 'best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    final_mae, final_mse = evaluator.evaluate(test_loader)
    print('Final MAE: {:.4f}, Final MSE: {:.4f}'.format(final_mae, final_mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=25)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--input_dir', type=str, default='data/nwpu')
    parser.add_argument('--output_dir', type=str, default='saved_nwpu')
    parser.add_argument('--resume', type=str, default='')
    # dataset config
    parser.add_argument('--num_clustering', type=int, default=4)
    parser.add_argument('--cluster_step', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    # model
    parser.add_argument('--arch', type=str, default='msMeta', choices=networks.names())
    # training config
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=135)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    print('Training dataset:', args.input_dir.split('/')[-1])
    setup_seed(args.seed)
    main(args)
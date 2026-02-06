import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from model import  Resnet50FPN, CountRegressor, weights_normal_init
from utils import MAPS, Scales, Transform, TransformTrain, extract_features
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import join
import random
import torch.optim as optim
import torch.nn.functional as F

def train(resnet50_conv, regressor, data_split, annotations, optimizer, criterion, best_mae, best_rmse, args):
    im_ids = data_split['train']
    random.shuffle(im_ids)
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    pbar = tqdm(im_ids)
    cnt = 0
    for im_id in pbar:
        cnt += 1
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])
        image = Image.open('{}/{}'.format(args.input_dir, im_id))
        image.load()
        density_path = os.path.join(args.gt_dir, im_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')    
        sample = {'image': image, 'lines_boxes': rects, 'gt_density': density}
        sample = TransformTrain(sample)
        image, boxes, gt_density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()
        with torch.no_grad():
            features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        features.requires_grad = True
        optimizer.zero_grad()
        output = regressor(features)
        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2], output.shape[3]), mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0:
                gt_density = gt_density * (orig_count / new_count)
        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(gt_density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        pbar.set_description('GT: {:.2f}, Pred: {:.2f}, Error: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}, Best val MAE: {:.2f}, Best val RMSE: {:.2f}'
                             .format( gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), train_mae / cnt, (train_rmse / cnt)**0.5, best_mae, best_rmse))
    train_loss = train_loss / len(im_ids)
    train_mae = (train_mae / len(im_ids))
    train_rmse = (train_rmse / len(im_ids))**0.5
    return train_loss, train_mae, train_rmse
   
def eval(resnet50_conv, regressor, data_split, annotations, args):
    cnt = 0
    SAE = 0
    SSE = 0
    im_ids = data_split[args.test_split]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])
        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])
        image = Image.open('{}/{}'.format(args.input_dir, im_id))
        image.load()
        sample = {'image': image, 'lines_boxes': rects}
        sample = Transform(sample)
        image, boxes = sample['image'].cuda(), sample['boxes'].cuda()
        with torch.no_grad():
            output = regressor(extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales))
        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2
        pbar.set_description('Image name: {:<8}, GT: {:2f}, {:.2f}, Error: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}'
                             .format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE / cnt, (SSE / cnt)**0.5))
    print('MAE: {:.2f}, RMSE: {:.2f}'.format(SAE / cnt, (SSE / cnt)**0.5))
    return SAE / cnt, (SSE / cnt)**0.5

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # loss
    criterion = nn.MSELoss().cuda()
    # model
    resnet50_conv = Resnet50FPN()
    resnet50_conv.cuda()
    resnet50_conv.eval()
    regressor = CountRegressor(6, pool='mean')
    weights_normal_init(regressor, dev=0.001)
    regressor.train()
    regressor.cuda()
    # optimizer
    optimizer = optim.Adam(regressor.parameters(), lr=args.lr)
    with open(args.ann_dir) as f:
        annotations = json.load(f)
    with open(args.split_dir) as f:
        data_split = json.load(f)
    best_mae, best_rmse = 1e7, 1e7
    stats = list()
    for epoch in range(0, args.epochs):
        regressor.train()
        train_loss, train_mae, train_rmse = train(resnet50_conv, regressor, data_split, annotations, optimizer, criterion, best_mae, best_rmse, args)
        regressor.eval()
        val_mae, val_rmse = eval(resnet50_conv, regressor, data_split, annotations, args)
        stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
        stats_file = join(args.output_dir, "stats" +  ".txt")
        with open(stats_file, 'w') as f:
            for s in stats:
                f.write("%s\n" % ','.join([str(x) for x in s]))
        if best_mae >= val_mae:
            best_mae = val_mae
            best_rmse = val_rmse
            model_name = os.path.join(args.output_dir, "FamNet_Save1.pth")
            torch.save(regressor.state_dict(), model_name)
        print('Epoch": [{}/{}], Loss: {:.2f}, Train MAE: {:.2f}, Train RMSE: {:.2f}, Val MAE: {:.2f}, Val RMSE: {:.2f}, Best val MAE: {:.2f},  Best val RMSE: {:.2f}'
              .format(epoch + 1, args.epochs, stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='FSC147')
    parser.add_argument("--input_dir", type=str, default='datasets/FSC147/images_384_VarV2')
    parser.add_argument('--gt_dir', type=str, default='datasets/FSC147/gt_density_map_adaptive_384_VarV2')
    parser.add_argument("--output_dir", type=str, default="saved_fsc147")
    parser.add_argument('--ann_dir', type=str, default='data/annotation_FSC147_384.json')
    parser.add_argument('--split_dir', type=str, default='data/Train_Test_Val_FSC_147.json')
    # training config
    parser.add_argument("--test_split", type=str, default='val', choices=["train", "test", "val"])
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    main(args)
import warnings
warnings.filterwarnings("ignore")
import copy
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features, MincountLoss, PerturbationLoss
from PIL import Image
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch.optim as optim

def main(args):
    # model
    resnet50_conv = Resnet50FPN()
    resnet50_conv.cuda()
    resnet50_conv.eval()
    regressor = CountRegressor(6, pool='mean')
    regressor.load_state_dict(torch.load(args.ckpt_dir))
    print('Load ckpt from:', args.ckpt_dir)
    regressor.cuda()
    regressor.eval()
    with open(args.ann_dir) as f:
        annotations = json.load(f)
    with open(args.split_dir) as f:
        data_split = json.load(f)
    cnt = 0
    SAE = 0
    SSE = 0
    im_ids = data_split[args.test_split]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points']) # [13, 2]
        rects = list()
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])
        image = Image.open('{}/{}'.format(args.input_dir, im_id))
        image.load()
        sample = {'image': image, 'lines_boxes': rects}
        sample = Transform(sample)
        image, boxes = sample['image'], sample['boxes']
        image = image.cuda() # [1, 384, 576]
        boxes = boxes.cuda() # [1, 3, 5]
        with torch.no_grad():
            features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales) # [1, 3, 6, 48, 72]
        if not args.adapt:
            with torch.no_grad():
                output = regressor(features) # [1, 1, 384, 576]
        else:
            features.required_grad = True
            adapted_regressor = copy.deepcopy(regressor)
            adapted_regressor.train()
            optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.lr)
            for step in range(0, args.gradient_steps):
                optimizer.zero_grad()
                output = adapted_regressor(features)
                lCount = args.weight_mincount * MincountLoss(output, boxes)
                lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
                Loss = lCount + lPerturbation
                if torch.is_tensor(Loss):
                    Loss.backward()
                    optimizer.step()
            features.required_grad = False
            output = adapted_regressor(features)
        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2
        pbar.set_description('Image name: {:<8}, GT: {:.2f}, Pred: {:.2f}, Error: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}'.format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE / cnt, (SSE / cnt)**0.5))
    print('MAE: {:.2f}, RMSE: {:.2f}'.format(SAE / cnt, (SSE / cnt)**0.5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='FSC147')
    parser.add_argument("--input_dir", type=str, default='datasets/FSC147/images_384_VarV2')
    parser.add_argument('--ann_dir', type=str, default='data/annotation_FSC147_384.json')
    parser.add_argument('--split_dir', type=str, default='data/Train_Test_Val_FSC_147.json')
    parser.add_argument("--test_split", type=str, default='val')
    parser.add_argument("--ckpt_dir", type=str, default="data/pretrainedModels/FamNet_Save1.pth")
    # testing config
    parser.add_argument("--adapt", action='store_true')
    parser.add_argument("--gradient_steps", type=int, default=100)
    parser.add_argument("-lr", type=float, default=1e-7)
    parser.add_argument("--weight_mincount", type=float, default=1e-9)
    parser.add_argument("--weight_perturbation", type=float, default=1e-4)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)
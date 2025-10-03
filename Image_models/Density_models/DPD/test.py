import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from torch.autograd import Variable
import misc.transforms as own_transforms
import tqdm
from model.locator import Crowd_locator
import numpy as np
import torchvision.transforms as standard_transforms
import torch
import os
from PIL import Image
import  cv2
from collections import OrderedDict
import argparse
torch.backends.cudnn.benchmark = True

def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)
    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'points': points}
    return pre_data, boxes

def test(file_list, args):
    # model
    net = Crowd_locator(args.net, args.gpu_id)
    net.cuda()
    state_dict = torch.load(args.ckpt_dir)['net']
    if len(args.gpu_id.split(',')) > 1:
        net.load_state_dict(state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    print('Load ckpt from:', args.ckpt_dir)
    net.eval()
    file_list = tqdm.tqdm(file_list)
    for infos in file_list:
        filename = infos.split()[0]
        imgname = os.path.join(args.input_dir, 'images', filename + '.jpg')
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        slice_h, slice_w = 512, 512
        slice_h, slice_w = slice_h, slice_w
        with torch.no_grad():
            img = Variable(img).cuda()
            b, c, h, w = img.shape
            crop_imgs, crop_dots, crop_masks = [], [], []
            if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                [pred_threshold, pred_map, __] = [i.cpu() for i in net(img, mask_gt=None, mode='val')]
            else:
                if h % 16 != 0:
                    pad_dims = (0, 0, 0, 16 - h % 16)
                    h = (h // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")
                if w % 16 != 0:
                    pad_dims = (0, 16 - w % 16, 0, 0)
                    w = (w // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")
                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                        mask = torch.zeros(1,1,img.size(2), img.size(3)).cpu()
                        mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = torch.cat(crop_imgs, dim=0), torch.cat(crop_masks, dim=0)
                crop_preds, crop_thresholds = [], []
                nz, period = crop_imgs.size(0), 4
                for i in range(0, nz, period):
                    [crop_threshold, crop_pred, __] = [i.cpu() for i in net(crop_imgs[i:min(nz, i+period)],mask_gt = None, mode='val')]
                    crop_preds.append(crop_pred)
                    crop_thresholds.append(crop_threshold)
                crop_preds = torch.cat(crop_preds, dim=0)
                crop_thresholds = torch.cat(crop_thresholds, dim=0)
                idx = 0
                pred_map = torch.zeros(b, 1, h, w).cpu()
                pred_threshold = torch.zeros(b, 1, h, w).cpu().float()
                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                        pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                        idx += 1
                mask = crop_masks.sum(dim=0)
                pred_map = (pred_map / mask)
                pred_threshold = (pred_threshold / mask)
            a = torch.ones_like(pred_map)
            b = torch.zeros_like(pred_map)
            binar_map = torch.where(pred_map >= pred_threshold, a, b)
            pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())
            out_file_name = os.path.join(args.output_dir, args.type_dataset + '_' + args.net + '_' + args.output_file)
            with open(out_file_name, 'a') as f:
                f.write(filename + ' ')
                f.write(str(pred_data['num']) + ' ')
                for ind,point in enumerate(pred_data['points'],1):
                    if ind < pred_data['num']:
                        f.write(str(int(point[0])) + ' ' + str(int(point[1])) + ' ')
                    else:
                        f.write(str(int(point[0])) + ' ' + str(int(point[1])))
                f.write('\n')
                f.close()

def main(args):
    txtpath = os.path.join(args.input_dir, args.output_file)
    with open(txtpath) as f:
        lines = f.readlines()
    test(lines, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SHHA', choices=['SHHA', 'SHHB', 'QNRF', 'NWPU', 'FDST', 'JHU'])
    parser.add_argument('--input_dir', type=str, default='data/SHHA')
    parser.add_argument('--output_dir', type=str, default='saved_sha_to_shb')
    parser.add_argument('--output_file', type=str, default='val.txt')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--net', type=str, default='VGG16_FPN', choices=['HR_Net', 'VGG16_FPN'])
    parser.add_argument('--ckpt_dir', type=str, default='saved_sha_to_shb')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    if args.type_dataset == 'NWPU':
        mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    elif args.type_dataset == 'SHHA':
        mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
    elif args.type_dataset == 'SHHB':
        mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    elif args.type_dataset == 'QNRF':
        mean_std = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])
    elif args.type_dataset == 'FDST':
        mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    elif args.type_dataset == 'JHU':
        mean_std = ([0.429683953524, 0.437104910612, 0.421978861094], [0.235549390316, 0.232568427920, 0.2355950474739])
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)])
    restore = standard_transforms.Compose([own_transforms.DeNormalize(*mean_std), standard_transforms.ToPILImage()])
    main(args)
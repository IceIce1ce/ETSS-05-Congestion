import warnings
warnings.filterwarnings('ignore')
from Networks.HR_Net.seg_hrnet import get_seg_model
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from image import load_data_fidt
import scipy
from utils import save_checkpoint, setup_seed
import cv2
import os
import numpy as np
import torch
import nni
from nni.utils import merge_parameter
from config import return_args, args

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pre_data(train_list, args, train):
    if train:
        print("Preload training for dataset:", args['type_dataset'])
    else:
        print('Preload testing for dataset:', args['type_dataset'])
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, fidt_map, kpoint = load_data_fidt(Img_path)
        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1
    return data_keys

def validate(Pre_data, model, args):
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, args['output_dir'], transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), args=args, train=False), batch_size=batch_size)
    model.eval()
    mae = 0.0
    mse = 0.0
    visi = []
    index = 0
    if args['write_loc']:
        if not os.path.exists(args['loc_dir']):
            os.makedirs(args['loc_dir'])
        if args['type_dataset'] == 'sha':
            f_loc = open(os.path.join(args['loc_dir'], 'A_localization.txt'), "w+")
        elif args['type_dataset'] == 'shb':
            f_loc = open(os.path.join(args['loc_dir'], 'B_localization.txt'), "w+")
        elif args['type_dataset'] == 'qnrf':
            f_loc = open(os.path.join(args['loc_dir'], 'qnrf_localization.txt'), "w+")
        elif args['type_dataset'] == 'jhu':
            f_loc = open(os.path.join(args['loc_dir'], 'jhu_localization.txt'), "w+")
        elif args['type_dataset'] == 'nwpu':
            f_loc = open(os.path.join(args['loc_dir'], 'nwpu_localization.txt'), "w+")
        elif args['type_dataset'] == 'trancos':
            f_loc = open(os.path.join(args['loc_dir'], 'trancos_localization.txt'), "w+")
        elif args['type_dataset'] == 'suwon':
            f_loc = open(os.path.join(args['loc_dir'], 'suwon_localization.txt'), "w+")
        else:
            print('This dataset does not exist')
            raise NotImplementedError
    else:
        f_loc = None
    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):
        count = 0
        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)
        with torch.no_grad():
            d6 = model(img)
            count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)
            if args['visual'] == True:
                zero_count = np.array(list(zip(np.nonzero(pred_kpoint)[1], np.nonzero(pred_kpoint)[0])))
                if zero_count.shape[0] != 0: # case none people
                    if not os.path.exists(os.path.join(args['output_dir'], 'vis_box')):
                        os.makedirs(os.path.join(args['output_dir'], 'vis_box'))
                    ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname, args)
                    show_fidt = show_map(d6.data.cpu().numpy())
                    gt_show = show_map(fidt_map.data.cpu().numpy())
                    res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                    cv2.imwrite(os.path.join(args['output_dir'], 'vis_box', fname[0]), res)
        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)
        if i % 1 == 0:
            print('File name: {}, GT: {:.2f}, Pred: {:.2f}'.format(fname[0], gt_count, count))
            visi.append([img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(), fname])
            index += 1
    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)
    nni.report_intermediate_result(mae)
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))
    return mae, visi

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()
    if args['type_dataset'] == 'qnrf':
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    else:
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1
    if input_max < 0.1:
        input = input * 0
    count = int(torch.sum(input).item())
    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()
    if f_loc != None:
        f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc

def generate_point_map(kpoint, f_loc, rate=1):
    pred_coor = np.nonzero(kpoint)
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)
    if f_loc != None:
        for data in coord_list:
            f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
        f_loc.write('\n')
    return point_map

def generate_bounding_boxes(kpoint, fname, args):
    if args['type_dataset'] == 'sha':
        Img_data = cv2.imread(os.path.join(args['input_dir'], 'part_A_final/test_data/images', fname[0]))
    elif args['type_dataset'] == 'shb':
        Img_data = cv2.imread(os.path.join(args['input_dir'], 'part_B_final/test_data/images', fname[0]))
    elif args['type_dataset'] == 'qnrf':
        Img_data = cv2.imread(os.path.join(args['input_dir'], 'test_data/images', fname[0]))
    elif args['type_dataset'] == 'jhu':
        Img_data = cv2.imread(os.path.join(args['input_dir'], 'test/images_2048', fname[0]))
    elif args['type_dataset'] == 'nwpu':
        Img_data = cv2.imread(os.path.join(args['input_dir'], 'images_2048', fname[0]))
    elif args['type_dataset'] == 'trancos':
        Img_data = cv2.imread(os.path.join(args['input_dir'], 'test_data/images', fname[0]))
    elif args['type_dataset'] == 'suwon':
        Img_data = cv2.imread(os.path.join(args['input_dir'], 'test_data/images', fname[0]))
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    ori_Img_data = Img_data.copy()
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2. # case 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)
        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)), (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)
    return ori_Img_data, Img_data

def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1

def main(args):
    if args['type_dataset'] == 'sha':
        test_file = 'npydata/ShanghaiA_test.npy'
    elif args['type_dataset'] == 'shb':
        test_file = 'npydata/ShanghaiB_test.npy'
    elif args['type_dataset'] == 'qnrf':
        test_file = 'npydata/qnrf_test.npy'
    elif args['type_dataset'] == 'jhu':
        test_file = 'npydata/jhu_test.npy'
    elif args['type_dataset'] == 'nwpu':
        test_file = 'npydata/nwpu_val.npy' # NWPU test does not have GT
    elif args['type_dataset'] == 'trancos':
        test_file = 'npydata/trancos_test.npy'
    elif args['type_dataset'] == 'suwon':
        test_file = 'npydata/suwon_test.npy'
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    # model
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    # optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args['lr']}])
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    # load pretrained model
    if args['pre']:
        if os.path.isfile(args['pre']):
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
            print('Load ckpt from:', args['pre'])
        else:
            print('No ckpt found at:', args['pre'])
    torch.set_num_threads(args['num_workers'])
    if args['preload_data'] == True:
        test_data = pre_data(val_list, args, train=False)
    else:
        test_data = val_list
    prec1, visi = validate(test_data, model, args)
    is_best = prec1 < args['best_pred']
    args['best_pred'] = min(prec1, args['best_pred'])
    save_checkpoint({'arch': args['pre'], 'state_dict': model.state_dict(), 'best_prec1': args['best_pred'], 'optimizer': optimizer.state_dict()}, visi, is_best, args['output_dir'])

if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    setup_seed(args.seed)
    main(params)
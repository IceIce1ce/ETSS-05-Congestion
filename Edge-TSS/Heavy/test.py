import os
from config import return_args, args
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import get_root_logger, setup_seed, save_checkpoint
import nni
from nni.utils import merge_parameter
import util.misc as utils
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
if args.backbone == 'resnet50' or args.backbone == 'resnet101':
    from Networks.CDETR import build_model
import warnings
warnings.filterwarnings('ignore')

def get_congestion_label(count):
    if count <= 24:
        return 0
    elif count <= 100:
        return 1
    else:
        return 2

def validate(Pre_data, model, args):
    # test loader
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, args['output_dir'], shuffle=False, transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), args=args, train=False), batch_size=1)
    model.eval()
    mae = 0.0
    mse = 0.0
    visi = []
    ### compute accuracy ###
    # pred_labels = []
    # gt_labels = []
    # correct_per_class = [0, 0, 0]
    # total_per_class = [0, 0, 0]
    ### compute accuracy ###
    for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)
        with torch.no_grad():
            img = img.cuda()
            outputs = model(img)
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), kpoint.shape[0] * args['num_queries'], dim=1)
        count = 0
        gt_count = torch.sum(kpoint).item()
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count[sub_count < args['threshold']] = 0
            sub_count[sub_count > 0] = 1
            sub_count = torch.sum(sub_count).item()
            count += sub_count
        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)
        ### compute accuracy ###
        # gt_label = get_congestion_label(gt_count)
        # gt_labels.append(gt_label)
        # pred_label = get_congestion_label(count)
        # pred_labels.append(pred_label)
        # total_per_class[gt_label] += 1
        # if pred_label == gt_label:
        #     correct_per_class[gt_label] += 1
        ### compute accuracy ###
        if i % 30 == 0:
            print('Name: {}, GT: {:.4f}, Pred: {:.4f}'.format(fname[0], gt_count, count))
    mae = mae / len(test_loader)
    mse = math.sqrt(mse / len(test_loader))
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))
    ### compute accuracy ###
    # overall_acc = sum(correct_per_class) / len(test_loader) * 100
    # print('Average accuracy: {:.2f}%'.format(overall_acc))
    # class_names = ["Low (0-24)", "Medium (25-100)", "High (> 100)"]
    # print('Per-class Congestion Accuracy:')
    # for i in range(3):
    #     if total_per_class[i] > 0:
    #         acc = correct_per_class[i] / total_per_class[i] * 100
    #         print('{}: {:.2f}% ({}/{} samples)'.format(class_names[i], acc, correct_per_class[i], total_per_class[i]))
    #     else:
    #         print('{}: N/A (0 samples)'.format(class_names[i]))
    ### compute accuracy ###
    return mae, mse, visi

def main(args):
    if args['type_dataset'] == 'jhu':
        test_file = 'npydata/jhu_val.npy'
    elif args['type_dataset'] == 'nwpu':
        test_file = 'npydata/nwpu_val.npy'
    elif args['type_dataset'] == 'trancos':
        test_file = 'npydata/trancos_test.npy'
    elif args['type_dataset'] == 'drone_vehicle':
        test_file = 'npydata/drone_vehicle_test.npy'
    elif args['type_dataset'] == 'suwon':
        test_file = 'npydata/suwon_test.npy'
    elif args['type_dataset'] == 'sub_suwon':
        test_file = 'npydata/sub_suwon_test.npy'
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()
    utils.init_distributed_mode(return_args)
    # model config
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[int(data) for data in list(args['gpu_id']) if data!=','])
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    logger = get_root_logger(args['output_dir'] + '/debug.log')
    writer = SummaryWriter(args['output_dir'])
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    logger.info("Number of params: {:.2f}M".format(num_params / 1e6))
    # optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args['lr']}], lr=args['lr'], weight_decay=args['weight_decay'])
    # resume training
    if args['pre']:
        if os.path.isfile(args['pre']):
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
            logger.info("Load ckpt from: {}".format(args['pre']))
        else:
            logger.info("No ckpt found")
    logger.info('Best result: {:.4f}'.format(args['best_pred']))
    torch.set_num_threads(args['num_workers'])
    eval_epoch = 0
    pred_mae, pred_mse, visi = validate(test_list, model, args)
    writer.add_scalar('Metrcis/MAE', pred_mae, eval_epoch)
    writer.add_scalar('Metrcis/MSE', pred_mse, eval_epoch)
    if args['save']:
        is_best = pred_mae < args['best_pred']
        args['best_pred'] = min(pred_mae, args['best_pred'])
        save_checkpoint({'arch': args['pre'], 'state_dict': model.state_dict(), 'best_prec1': args['best_pred'], 'optimizer': optimizer.state_dict()}, visi, is_best, args['output_dir'])
    if args['local_rank'] == 0:
        logger.info('MAE: {:.4f}, MSE: {:.4f}, Best MAE: {:.4f}'.format(args['epochs'], pred_mae, pred_mse, args['best_pred']))

if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    print('Testing dataset:', params['type_dataset'])
    setup_seed(args.seed)
    main(params)
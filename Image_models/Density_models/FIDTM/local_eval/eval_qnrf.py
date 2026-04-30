import os
import numpy as np
from scipy import spatial as ss
from utils import hungarian, read_pred_and_gt, AverageMeter, AverageCategoryMeter
import argparse
import scipy.io
import math

def compute_metrics(dist_matrix, match_matrix, pred_num, sigma, level, args):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma
    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]
    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]
    tp_c = np.zeros([args.num_classes])
    fn_c = np.zeros([args.num_classes])
    for i_class in range(args.num_classes):
        tp_c[i_class] = (level[tp_gt_index] == i_class).sum()
        fn_c[i_class] = (level[fn_gt_index] == i_class).sum()
    return tp, fp, fn, tp_c, fn_c

def main(gt_file, pred_file, args):
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
    metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(args.num_classes), 'fn_c': AverageCategoryMeter(args.num_classes)}
    pred_data, gt_data = read_pred_and_gt(pred_file, gt_file)
    for i_sample in id_std:
        tp_s, fp_s, fn_s, tp_l, fp_l, fn_l = [0, 0, 0, 0, 0, 0]
        tp_c_l = np.zeros([args.num_classes])
        fn_c_l = np.zeros([args.num_classes])
        if gt_data[i_sample]['num'] == 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            fp_pred_index = np.array(range(pred_p.shape[0]))
            fp_l = fp_pred_index.shape[0]
        if pred_data[i_sample]['num'] == 0 and gt_data[i_sample]['num'] != 0:
            gt_p = gt_data[i_sample]['points']
            level = gt_data[i_sample]['level']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            fn_l = fn_gt_index.shape[0]
            for i_class in range(args.num_classes):
                fn_c_l[i_class] = (level[fn_gt_index] == i_class).sum()
        if gt_data[i_sample]['num'] != 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            gt_p = gt_data[i_sample]['points']
            sigma_l = gt_data[i_sample]['sigma'][:, 1]
            level = gt_data[i_sample]['level']
            dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
            tp_l, fp_l, fn_l, tp_c_l, fn_c_l = compute_metrics(dist_matrix, match_matrix, pred_p.shape[0], sigma_l, level, args)
        metrics_l['tp'].update(tp_l)
        metrics_l['fp'].update(fp_l)
        metrics_l['fn'].update(fn_l)
        metrics_l['tp_c'].update(tp_c_l)
        metrics_l['fn_c'].update(fn_c_l)
        gt_count, pred_cnt = gt_data[i_sample]['num'], pred_data[i_sample]['num']
        s_mae = abs(gt_count - pred_cnt)
        s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)
        cnt_errors['mae'].update(s_mae)
        cnt_errors['mse'].update(s_mse)
        if gt_count != 0:
            s_nae = abs(gt_count - pred_cnt) / gt_count
            cnt_errors['nae'].update(s_nae)
    ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
    ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
    f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l)
    print('AP large: {:.2f}, AR large: {:.2f}, F1m large: {:.2f}'.format(ap_l, ar_l, f1m_l))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='qnrf')
    parser.add_argument('--input_dir', default= '../datasets/UCF-QNRF', type=str)
    parser.add_argument('--num_classes', type=int, default=1)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    id_std = [i for i in range(1, 335, 1)]
    img_test_path = os.path.join(args.input_dir, 'Test/')
    gt_test_path = os.path.join(args.input_dir, 'Test/')
    img_test = []
    gt_test = []
    for file_name in os.listdir(img_test_path):
        if file_name.split('.')[1] == 'jpg':
            img_test.append(img_test_path + file_name)
    for file_name in os.listdir(gt_test_path):
        if file_name.split('.')[1] == 'mat':
            gt_test.append(file_name)
    img_test.sort()
    gt_test.sort()
    ap_l_list = []
    ar_l_list = []
    f1m_l_list = []
    for i in range(1, 101):
        f = open('point_files/qnrf_gt.txt', 'w+')
        k = 0
        for img_path in img_test:
            Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])
            Gt_data = Gt_data['annPoints']
            f.write('{} {} '.format(k + 1, len(Gt_data)))
            sigma_s = 4
            sigma_l = i
            for data in Gt_data:
                f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
            f.write('\n')
            k = k + 1
        f.close()
        gt_file = 'point_files/qnrf_gt.txt'
        pred_file = 'point_files/qnrf_pred_fidt.txt'
        main(gt_file, pred_file, args)
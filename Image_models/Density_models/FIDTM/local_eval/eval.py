import numpy as np
from scipy import spatial as ss
from utils import hungarian, read_pred_and_gt, AverageMeter, AverageCategoryMeter
import argparse

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

def main(args):
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter(), }
    metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(args.num_classes), 'fn_c': AverageCategoryMeter(args.num_classes)}
    metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(args.num_classes), 'fn_c': AverageCategoryMeter(args.num_classes)}
    pred_data, gt_data = read_pred_and_gt(pred_file, gt_file)
    for i_sample in id_std:
        tp_s, fp_s, fn_s, tp_l, fp_l, fn_l = [0, 0, 0, 0, 0, 0]
        tp_c_s = np.zeros([args.num_classes])
        fn_c_s = np.zeros([args.num_classes])
        tp_c_l = np.zeros([args.num_classes])
        fn_c_l = np.zeros([args.num_classes])
        if gt_data[i_sample]['num'] == 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            fp_pred_index = np.array(range(pred_p.shape[0]))
            fp_s = fp_pred_index.shape[0]
            fp_l = fp_pred_index.shape[0]
        if pred_data[i_sample]['num'] == 0 and gt_data[i_sample]['num'] != 0:
            gt_p = gt_data[i_sample]['points']
            level = gt_data[i_sample]['level']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            fn_s = fn_gt_index.shape[0]
            fn_l = fn_gt_index.shape[0]
            for i_class in range(args.num_classes):
                fn_c_s[i_class] = (level[fn_gt_index] == i_class).sum()
                fn_c_l[i_class] = (level[fn_gt_index] == i_class).sum()
        if gt_data[i_sample]['num'] != 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            gt_p = gt_data[i_sample]['points']
            sigma_s = gt_data[i_sample]['sigma'][:, 0]
            sigma_l = gt_data[i_sample]['sigma'][:, 1]
            level = gt_data[i_sample]['level']
            dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
            tp_s, fp_s, fn_s, tp_c_s, fn_c_s = compute_metrics(dist_matrix, match_matrix, pred_p.shape[0], sigma_s, level, args)
            tp_l, fp_l, fn_l, tp_c_l, fn_c_l = compute_metrics(dist_matrix, match_matrix, pred_p.shape[0], sigma_l, level, args)
        metrics_s['tp'].update(tp_s)
        metrics_s['fp'].update(fp_s)
        metrics_s['fn'].update(fn_s)
        metrics_s['tp_c'].update(tp_c_s)
        metrics_s['fn_c'].update(fn_c_s)
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
    ap_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fp'].sum + 1e-20)
    ar_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fn'].sum + 1e-20)
    f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s)
    ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
    ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
    f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l)
    print('AP small: {:.2f}, AR small: {:.2f}, F1m small: {:.2f}, AP large: {:.2f}, AR large: {:.2f}, F1m large: {:.2f}'.format(ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l))
    mae = cnt_errors['mae'].avg
    mse = np.sqrt(cnt_errors['mse'].avg)
    nae = cnt_errors['nae'].avg
    print('MAE: {:.2f}, MSE: {:.2f}, NAE: {:.2f}'.format(mae, mse, nae))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--num_classes', type=int, default=1)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    if args.type_dataset == 'sha':
        gt_file = 'point_files/A_gt.txt'
        pred_file = 'point_files/A_pred_fidt.txt'
        id_std = [i for i in range(1, 183, 1)]
    elif args.type_dataset == 'shb':
        gt_file = 'point_files/B_gt.txt'
        pred_file = 'point_files/B_pred_fidt.txt'
        id_std = [i for i in range(1, 317, 1)]
    elif args.type_dataset == 'jhu':
        gt_file = 'point_files/jhu_gt.txt'
        pred_file = 'point_files/jhu_pred_fidt.txt'
        id_std = [i for i in range(1, 1601, 1)]
    elif args.type_dataset == 'nwpu':
        gt_file = 'point_files/nwpu_gt (val set).txt'
        pred_file = 'point_files/nwpu_pred_fidt (val set).txt'
        id_std = [i for i in range(3110, 3610, 1)]
        id_std[59] = 3098
    main(args)
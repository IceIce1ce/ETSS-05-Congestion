import os
import numpy as np
from scipy import spatial as ss
import cv2
from misc.utils import hungarian, read_pred_and_gt
import argparse

def main(args):
    pred_data, gt_data = read_pred_and_gt(pred_file, gt_file)
    for i_sample in id_std:
        # if i_sample not in gt_data or i_sample not in pred_data:
        #     print(f"Warning: Sample {i_sample} missing in {'gt_data' if i_sample not in gt_data else 'pred_data'}")
        #     continue
        gt_p, pred_p, fn_gt_index, tp_pred_index, fp_pred_index, ap, ar = [], [], [], [], [], [], []
        if gt_data[i_sample]['num'] == 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
        if pred_data[i_sample]['num'] ==0 and gt_data[i_sample]['num'] != 0:
            gt_p = gt_data[i_sample]['points']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            sigma_l = gt_data[i_sample]['sigma'][:, 1]
        if gt_data[i_sample]['num'] != 0 and pred_data[i_sample]['num'] != 0:
            pred_p =  pred_data[i_sample]['points']
            gt_p = gt_data[i_sample]['points']
            sigma_l = gt_data[i_sample]['sigma'][:, 1]
            dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
            match_matrix = np.zeros(dist_matrix.shape,dtype=bool)
            for i_pred_p in range(pred_p.shape[0]):
                pred_dist = dist_matrix[i_pred_p,:]
                match_matrix[i_pred_p,:] = pred_dist<=sigma_l
            tp, assign = hungarian(match_matrix)
            fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
            tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
            fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]
            pre = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fp_pred_index.shape[0] + 1e-20)
            rec = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fn_gt_index.shape[0] + 1e-20)
        if args.type_dataset == 'FDST':
            sample_str = str(i_sample)
            formatted_sample = f"{sample_str[:-3]}_{sample_str[-3:]}"
            img = cv2.imread(img_path + '/' + formatted_sample + '.jpg')
        else:
            img = cv2.imread(img_path + '/' + str(i_sample).zfill(4) + '.jpg')
        point_r_value = 5
        thickness = 3
        if gt_data[i_sample]['num'] != 0:
            for i in range(gt_p.shape[0]):
                if i in fn_gt_index:
                    cv2.circle(img, (gt_p[i][0], gt_p[i][1]), point_r_value, (0, 0, 255), -1) # red
                    cv2.circle(img, (gt_p[i][0], gt_p[i][1]), sigma_l[i], (0, 0, 255), thickness)
                else:
                    cv2.circle(img, (gt_p[i][0], gt_p[i][1]), sigma_l[i], (0, 255, 0), thickness) # green
        if pred_data[i_sample]['num'] != 0:
            for i in range(pred_p.shape[0]):
                if i in tp_pred_index:
                    cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value, (0, 255, 0), -1) # green
                else:
                    cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value * 2, (255, 0, 255), -1) # magenta
        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
        cv2.imwrite(os.path.join(args.vis_dir, str(i_sample) + '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] + '.jpg'), img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SHHA', choices=['SHHA', 'SHHB', 'QNRF', 'NWPU', 'FDST', 'JHU'])
    parser.add_argument('--input_dir', type=str, default='data/SHHA')
    parser.add_argument('--output_dir', type=str, default='saved_sha_to_shb')
    parser.add_argument('--net', type=str, default='VGG16_FPN', choices=['HR_Net', 'VGG16_FPN'])
    parser.add_argument('--output_file', type=str, default='val.txt')
    parser.add_argument('--vis_dir', type=str, default='vis_sha_to_shb')
    args = parser.parse_args()

    print('Visualize for dataset:', args.type_dataset)
    gt_file = os.path.join(args.input_dir, 'val_gt_loc.txt')
    img_path = ori_data = os.path.join(args.input_dir, 'images')
    pred_file = os.path.join(args.output_dir, args.type_dataset + '_' + args.net + '_' + args.output_file)
    if args.type_dataset == 'NWPU':
        id_std = [i for i in range(3110, 3610, 1)]
        id_std[59] = 3098
    else:
        val_file = os.path.join(args.input_dir, 'val.txt')
        with open(val_file, 'r') as f:
            id_std = [int(line.strip()) for line in f]
    main(args)
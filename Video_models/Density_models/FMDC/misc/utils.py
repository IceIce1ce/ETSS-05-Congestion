import os
import cv2
from PIL import Image
import torch
from .flow_viz import flow_to_image
import numpy as np

def adjust_learning_rate(optimizer, epoch,base_lr1=0, base_lr2=0, power=0.9):
    lr1 =  base_lr1 * power ** ((epoch-1))
    lr2 =  base_lr2 * power ** ((epoch - 1))
    optimizer.param_groups[0]['lr'] = lr1
    optimizer.param_groups[1]['lr'] = lr2
    return lr1 , lr2

def save_results_color(iter, exp_path, restore, img0, img1, mask):
    UNIT_H , UNIT_W = img0.size(2), img0.size(3)
    for idx, tensor in enumerate(zip(img0.cpu().data, img1.cpu().data, mask)):
        if idx > 1:
            break
        pil_input0 = restore(tensor[0])
        pil_input1 = restore(tensor[1])
        pil_input0 = np.array(pil_input0)
        pil_input1 = np.array(pil_input1)
        pil_pred = np.copy(pil_input1)
        pil_pred[:, :, 1:] = mask
        pil_input0 = Image.fromarray(cv2.cvtColor(pil_input0, cv2.COLOR_LAB2RGB))
        pil_input1 = Image.fromarray(cv2.cvtColor(pil_input1, cv2.COLOR_LAB2RGB))
        pil_pred = Image.fromarray(cv2.cvtColor(pil_pred, cv2.COLOR_LAB2RGB))
        imgs = [pil_input0, pil_input1, pil_pred]
        w_num , h_num = 3, 1
        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            x, y = int(count%w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)
            target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
            count+=1
        target.save(os.path.join(exp_path,'{}_color.jpg'.format(iter)))

def save_results_mask(iter, exp_path, restore, img0, img1, pred_map0, gt_map0, pred_map1, gt_map1, pred_mask_out, gt_mask_out, pred_mask_in, gt_mask_in, f_flow,b_flow):
    UNIT_H , UNIT_W = img0.size(2), img0.size(3)
    for idx, tensor in enumerate(zip(img0.cpu().data, img1.cpu().data,pred_map0, gt_map0, pred_map1, gt_map1, pred_mask_out, gt_mask_out, pred_mask_in, gt_mask_in)):
        if idx > 1:
            break
        pil_input0 = restore(tensor[0])
        pil_input1 = restore(tensor[1])
        f_flow_map = flow_to_image(f_flow)
        b_flow_map = flow_to_image(b_flow)
        pred_color_map0 = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map0 = cv2.applyColorMap((255 * tensor[3] / (tensor[3].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        pred_color_map1 = cv2.applyColorMap((255 * tensor[4] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map1 = cv2.applyColorMap((255 * tensor[5] / (tensor[5].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        pred_out_color = cv2.applyColorMap((255 * tensor[6] / (tensor[6].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_out_color = cv2.applyColorMap((255 * tensor[7]/ (tensor[7].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        pred_in_color = cv2.applyColorMap((255 * tensor[8] / (tensor[8].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_in_color = cv2.applyColorMap((255 * tensor[9]/ (tensor[9].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        pil_input0 = np.array(pil_input0)
        pil_input1 = np.array(pil_input1)
        pil_input0 = Image.fromarray(cv2.cvtColor(pil_input0, cv2.COLOR_LAB2RGB))
        pil_input1 = Image.fromarray(cv2.cvtColor(pil_input1, cv2.COLOR_LAB2RGB))
        pil_output0 = Image.fromarray(cv2.cvtColor(pred_color_map0, cv2.COLOR_BGR2RGB))
        pil_gt0 = Image.fromarray(cv2.cvtColor(gt_color_map0, cv2.COLOR_BGR2RGB))
        pil_output1 = Image.fromarray(cv2.cvtColor(pred_color_map1, cv2.COLOR_BGR2RGB))
        pil_gt1 = Image.fromarray(cv2.cvtColor(gt_color_map1, cv2.COLOR_BGR2RGB))
        pil_maskout = Image.fromarray(cv2.cvtColor(pred_out_color, cv2.COLOR_BGR2RGB))
        pil_gtmaskout = Image.fromarray(cv2.cvtColor(gt_out_color, cv2.COLOR_BGR2RGB))
        pil_maskin = Image.fromarray(cv2.cvtColor(pred_in_color, cv2.COLOR_BGR2RGB))
        pil_gtmaskin = Image.fromarray(cv2.cvtColor(gt_in_color, cv2.COLOR_BGR2RGB))
        pil_f_flow = Image.fromarray(f_flow_map)
        pil_b_blow = Image.fromarray(b_flow_map)
        imgs = [pil_input0, pil_gt0, pil_output0, pil_f_flow,pil_gtmaskout, pil_maskout, pil_input1, pil_gt1, pil_output1, pil_b_blow,pil_gtmaskin, pil_maskin]
        w_num , h_num = 6, 2
        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            x, y = int(count % w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)
            target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
            count+=1
        if not os.path.exists(os.path.join(exp_path, 'onlymean_offset')):
            os.makedirs(os.path.join(exp_path, 'onlymean_offset'))
        target.save(os.path.join(exp_path,'onlymean_offset','{}_den.jpg'.format(iter)))

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    content = '  ['
    for key, data in scores.items():
        if isinstance(data, str):
            content +=(' ' + key + ' %s' % data)
        else:
            content += (' ' + key + ' %.3f' % data)
    content += ']'
    print(content)
    print(' ' + '-' * 20)
    best_str = '[best]'
    for key, data in train_record.items():
        best_str += (' [' + key +' %s'% data + ']')
    print(best_str)
    print('=' * 50)

def update_model(trainer, scores):
    train_record = trainer.train_record
    epoch = trainer.epoch
    snapshot_name = 'ep_%d_iter_%d' % (epoch, trainer.i_tb)
    for key, data in scores.items():
        snapshot_name += ('_'+ key+'_%.3f' % data)
    for key, data in  scores.items():
        if data < train_record[key]:
            train_record['best_model_name'] = snapshot_name
            to_saved_weight = trainer.net.state_dict()
            torch.save(to_saved_weight, os.path.join(trainer.output_dir, snapshot_name + '.pth'))
        if data < train_record[key]:
            train_record[key] = data
    latest_state = {'train_record': train_record, 'net': trainer.net.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'epoch': trainer.epoch, 'i_tb': trainer.i_tb}
    torch.save(latest_state,os.path.join(trainer.output_dir, 'latest_state.pth'))
    return train_record

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count
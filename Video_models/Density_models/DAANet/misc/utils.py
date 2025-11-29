import os
import torch
from .flow_viz import flow_to_image
import cv2
from PIL import Image
import numpy as np

def adjust_learning_rate(optimizer, epoch, base_lr1=0, base_lr2=0, power=0.9):
    lr1 = base_lr1 * power**((epoch - 1))
    lr2 = base_lr2 * power**((epoch - 1))
    optimizer.param_groups[0]['lr'] = lr1
    optimizer.param_groups[1]['lr'] = lr2
    optimizer.param_groups[2]['lr'] = lr2
    optimizer.param_groups[3]['lr'] = lr1
    return lr1 , lr2

def save_results_mask(args, scene_name, iter, restore, batch, img0, img1, den0, den1, out_map, in_map, gt_io_map, attn0, attn1, f_flow, b_flow, den_scales, gt_den_scales, mask, gt_mask):
    UNIT_H , UNIT_W = img0.size(2), img0.size(3)
    gaussian_kernel = 31
    gaussian_sigma = 10
    if args.mode == 'test':
        args.train_batch_size = args.val_batch_size
    COLOR_MAP_ATTN = [[255, 255, 0], [255, 0, 255], [0, 255, 255]]
    COLOR_MAP_ATTN = np.array(COLOR_MAP_ATTN, dtype="uint8") # [3, 3]
    for idx, tensor in enumerate(zip(img0.cpu().data, img1.cpu().data, den0, den1, out_map, in_map, gt_io_map, attn0, attn1)):
        if idx > 1:
            break
        f_flow_map = []
        b_flow_map = []
        den_scales_1_map = []
        gt_den_scales_1_map = []
        den_scales_2_map = []
        gt_den_scales_2_map = []
        attn_map_scale_1 = []
        attn_map_scale_2 = []
        a = [0, 0, 0]
        pil_input0 = restore(tensor[0]) # [768, 1024, 3]
        pil_input1 = restore(tensor[1]) # [768, 1024, 3]
        for i in range(len(den_scales)):
            f = f_flow[i][batch].permute(1, 2, 0).detach().cpu().numpy() # [768, 1024, 72]
            b = b_flow[i][batch].permute(1, 2, 0).detach().cpu().numpy() # [768, 1024, 72]
            f = cv2.resize(flow_to_image(f), (UNIT_W, UNIT_H)) # [768, 1024, 3]
            b = cv2.resize(flow_to_image(b), (UNIT_W, UNIT_H)) # [768, 1024, 3]
            f_flow_map.append(Image.fromarray(f))
            b_flow_map.append(Image.fromarray(b))
            den_scale_1 = den_scales[i][0].detach().cpu().numpy()[0]
            den_scale_2 = den_scales[i][1].detach().cpu().numpy()[0]
            gt_den_scale_1 = gt_den_scales[i][0].detach().cpu().numpy()[0]
            gt_den_scale_2 = gt_den_scales[i][1].detach().cpu().numpy()[0]
            den_scale_1 = cv2.GaussianBlur(den_scale_1, (int(gaussian_kernel / 2**i + a[i]), int(gaussian_kernel / 2**i + a[i]),), int(10 / 2**i))
            den_scale_2 = cv2.GaussianBlur(den_scale_2, (int(gaussian_kernel / 2**i + a[i]), int(gaussian_kernel / 2**i+a[i]),), int(10 / 2**i))
            gt_den_scale_1 = cv2.GaussianBlur(gt_den_scale_1, (int(gaussian_kernel / 2**i + a[i]), int(gaussian_kernel / 2**i + a[i]),),int(10 / 2**i))
            gt_den_scale_2 = cv2.GaussianBlur(gt_den_scale_2, (int(gaussian_kernel / 2**i + a[i]), int(gaussian_kernel / 2**i + a[i]),),int(10 / 2**i))
            den_scale_1 = cv2.resize(cv2.applyColorMap((255 * den_scale_1 / (den_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            den_scale_2 = cv2.resize(cv2.applyColorMap((255 * den_scale_2 / (den_scale_2.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            gt_den_scale_1 = cv2.resize(cv2.applyColorMap((255 * gt_den_scale_1 / (gt_den_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            gt_den_scale_2 = cv2.resize(cv2.applyColorMap((255 * gt_den_scale_2 / (gt_den_scale_2.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
            den_scale_1 = cv2.cvtColor(den_scale_1, cv2.COLOR_BGR2RGB)
            den_scale_2 = cv2.cvtColor(den_scale_2, cv2.COLOR_BGR2RGB)
            gt_den_scale_1 = cv2.cvtColor(gt_den_scale_1, cv2.COLOR_BGR2RGB)
            gt_den_scale_2 = cv2.cvtColor(gt_den_scale_2, cv2.COLOR_BGR2RGB)
            den_scales_1_map.append(Image.fromarray(den_scale_1))
            den_scales_2_map.append(Image.fromarray(den_scale_2))
            gt_den_scales_1_map.append(Image.fromarray(gt_den_scale_1))
            gt_den_scales_2_map.append(Image.fromarray(gt_den_scale_2))
            # attention map
            attn_1 = cv2.resize(cv2.applyColorMap((255 * tensor[7][i]).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
            attn_2 = cv2.resize(cv2.applyColorMap((255 * tensor[8][i]).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
            attn_1 = Image.fromarray(cv2.cvtColor(attn_1, cv2.COLOR_BGR2RGB))
            attn_2 = Image.fromarray(cv2.cvtColor(attn_2, cv2.COLOR_BGR2RGB))
            attn_map_scale_1.append(attn_1)
            attn_map_scale_2.append(attn_2)
            tensor[7][i] = cv2.GaussianBlur(tensor[7][i], (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
            tensor[8][i] = cv2.GaussianBlur(tensor[8][i], (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
        den0_map = cv2.resize(cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        den1_map = cv2.resize(cv2.applyColorMap((255 * tensor[3] / (tensor[3].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        # mask
        mask_out = mask[0, :, :, :].detach().cpu().numpy()
        mask_in=  mask[args.train_batch_size, :, :, :].detach().cpu().numpy()
        gt_mask_out= gt_mask[0, 0:1, :, :].detach().cpu().numpy()
        gt_mask_in = gt_mask[0, 1:2, :, :].detach().cpu().numpy()
        mask_out = cv2.GaussianBlur(mask_out[0], (gaussian_kernel, gaussian_kernel,), gaussian_sigma)
        mask_in = cv2.GaussianBlur(mask_in[0], (gaussian_kernel, gaussian_kernel,), gaussian_sigma)
        gt_mask_out = cv2.GaussianBlur(gt_mask_out[0], (gaussian_kernel, gaussian_kernel,), gaussian_sigma)
        gt_mask_in = cv2.GaussianBlur(gt_mask_in[0], (gaussian_kernel, gaussian_kernel,), gaussian_sigma)
        gt_mask_out[gt_mask_out > 0.15] = 1
        gt_mask_in[gt_mask_in > 0.15] = 1
        mask_out = cv2.resize(cv2.applyColorMap((255 * mask_out / (mask_out.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H)) 
        mask_in = cv2.resize(cv2.applyColorMap((255 * mask_in / (mask_in.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H)) 
        gt_mask_out = cv2.resize(cv2.applyColorMap((255 * gt_mask_out / (gt_mask_out.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H)) 
        gt_mask_in = cv2.resize(cv2.applyColorMap((255 * gt_mask_in / (gt_mask_in.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H))
        # io density map
        out_map = cv2.resize(cv2.applyColorMap((255 * tensor[4] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        in_map = cv2.resize(cv2.applyColorMap((255 * tensor[5] / (tensor[5].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        gt_out_map = cv2.resize(cv2.applyColorMap((255 * tensor[6][0] / (tensor[6][0].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        gt_in_map = cv2.resize(cv2.applyColorMap((255 * tensor[6][1] / (tensor[6][1].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        # argmax attention map
        attn_map0 = np.argmax(tensor[7], axis=0)
        attn_map0 = cv2.resize(COLOR_MAP_ATTN[attn_map0].squeeze(),  (UNIT_W, UNIT_H))
        attn_map0_dot = 255 - attn_map0 * np.repeat(((gt_den_scales[0][0].detach().cpu().numpy())>0.05).squeeze(),3,axis=1).reshape(UNIT_H, UNIT_W, 3)
        attn_map1 = np.argmax(tensor[8], axis=0)
        attn_map1 = cv2.resize(COLOR_MAP_ATTN[attn_map1].squeeze(),  (UNIT_W, UNIT_H))
        attn_map1_dot = 255 - attn_map1 * np.repeat((((gt_den_scales[0][1].detach().cpu().numpy())>0.05)).squeeze(),3,axis=1).reshape(UNIT_H, UNIT_W, 3)
        # mean offset map
        f_flow_map_arrays = [np.array(img) for img in f_flow_map]
        b_flow_map_arrays = [np.array(img) for img in b_flow_map]
        f_flow_map_m = np.mean(f_flow_map_arrays, axis=0)
        b_flow_map_m = np.mean(b_flow_map_arrays, axis=0)
        pil_input0 = np.array(pil_input0)
        pil_input1 = np.array(pil_input1)
        pil_input0 = Image.fromarray(pil_input0)
        pil_input1 = Image.fromarray(pil_input1)
        den0_map = Image.fromarray(cv2.cvtColor(den0_map, cv2.COLOR_BGR2RGB))
        den1_map = Image.fromarray(cv2.cvtColor(den1_map, cv2.COLOR_BGR2RGB))
        mask_out = Image.fromarray(cv2.cvtColor(mask_out, cv2.COLOR_BGR2RGB))
        mask_in = Image.fromarray(cv2.cvtColor(mask_in, cv2.COLOR_BGR2RGB))
        gt_mask_out = Image.fromarray(cv2.cvtColor(gt_mask_out, cv2.COLOR_BGR2RGB))
        gt_mask_in = Image.fromarray(cv2.cvtColor(gt_mask_in, cv2.COLOR_BGR2RGB))
        out_map = Image.fromarray(cv2.cvtColor(out_map, cv2.COLOR_BGR2RGB))
        in_map = Image.fromarray(cv2.cvtColor(in_map, cv2.COLOR_BGR2RGB))
        gt_out_map = Image.fromarray(cv2.cvtColor(gt_out_map, cv2.COLOR_BGR2RGB))
        gt_in_map = Image.fromarray(cv2.cvtColor(gt_in_map, cv2.COLOR_BGR2RGB))
        attn_map0 = Image.fromarray(cv2.cvtColor(attn_map0, cv2.COLOR_BGR2RGB))
        attn_map1 = Image.fromarray(cv2.cvtColor(attn_map1, cv2.COLOR_BGR2RGB))
        attn_map0_dot = Image.fromarray(cv2.cvtColor(attn_map0_dot, cv2.COLOR_BGR2RGB))
        attn_map1_dot = Image.fromarray(cv2.cvtColor(attn_map1_dot, cv2.COLOR_BGR2RGB))
        f_flow_map_m = Image.fromarray(np.uint8(f_flow_map_m))
        b_flow_map_m = Image.fromarray(np.uint8(b_flow_map_m))
        imgs = [pil_input0, out_map, gt_out_map, f_flow_map_m, mask_out, gt_mask_out, den0_map, attn_map0, attn_map0_dot, attn_map_scale_1[2], attn_map_scale_1[1], attn_map_scale_1[0],
                f_flow_map[2], f_flow_map[1], f_flow_map[0], den_scales_1_map[2], den_scales_1_map[1], den_scales_1_map[0], gt_den_scales_1_map[2], gt_den_scales_1_map[1],
                gt_den_scales_1_map[0], pil_input1, in_map, gt_in_map, b_flow_map_m, mask_in, gt_mask_in, den1_map, attn_map1, attn_map1_dot, attn_map_scale_2[2], attn_map_scale_2[1],
                attn_map_scale_2[0], b_flow_map[2], b_flow_map[1], b_flow_map[0], den_scales_2_map[2], den_scales_2_map[1], den_scales_2_map[0], gt_den_scales_2_map[2],
                gt_den_scales_2_map[1], gt_den_scales_2_map[0]]
        w_num , h_num = 3, 16
        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            x, y = int(count % w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)
            target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
            count += 1
        if args.mode == 'test':
            try:    
                dir = os.path.join(args.output_dir, scene_name.split('/')[-1])
            except:
                dir = './'
        else:
            dir = os.path.join(args.output_dir, 'vis')
        if not os.path.exists(dir):
            os.makedirs(dir)
        target.resize((w_num * 50, h_num * 50)) # [12448, 3102, 3]
        target.save(os.path.join(dir, f'{iter}_{batch}_den.jpg'.format()))

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    content = ' ['
    for key, data in scores.items():
        if isinstance(data,str):
            content += (' ' + key + ' %s' % data)
        else:
            content += (' ' + key + ' %.2f' % data)
    content += ']'
    print(content)
    print(' '+ '-' * 20)
    best_str = '[best]'
    for key, data in train_record.items():
        best_str += ('[' + key +' %s'% data + ']')
    print(best_str)

def update_model(trainer, scores, val=False):
    train_record = trainer.train_record
    if val:
        epoch = trainer.epoch
        snapshot_name = 'ep_%d_iter_%d'% (epoch, trainer.i_tb)
        for key, data in scores.items():
            snapshot_name += ('_' + key + '_%.3f' % data)
        for key, data in  scores.items():
            if data < train_record[key]:
                train_record['best_model_name'] = snapshot_name
                to_saved_weight = trainer.net.state_dict()
                torch.save(to_saved_weight, os.path.join(trainer.output_dir, snapshot_name + '.pth'))
            if data < train_record[key]:
                train_record[key] = data
    latest_state = {'train_record': train_record, 'net': trainer.net.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'epoch': trainer.epoch, 'i_tb':trainer.i_tb,
                    'output_dir': trainer.output_dir, 'args': trainer.args}
    torch.save(latest_state, os.path.join(trainer.output_dir, 'latest_state.pth'))
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
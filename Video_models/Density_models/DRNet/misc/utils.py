import os
import numpy as np
import cv2
from PIL import Image
import torch

def adjust_learning_rate(optimizer, epoch,base_lr1=0, base_lr2=0, power=0.9):
    lr1 =  base_lr1 * power ** ((epoch-1))
    lr2 =  base_lr2 * power ** ((epoch - 1))
    optimizer.param_groups[0]['lr'] = lr1
    optimizer.param_groups[1]['lr'] = lr2
    return lr1 , lr2

def save_results_more(iter, exp_path, restore, img, pred_map, gt_map, binar_map, threshold_matrix, Instance_weights):
    UNIT_H , UNIT_W = img.size(2), img.size(3)
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, binar_map, threshold_matrix,Instance_weights)):
        if idx > 1:
            break
        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_matched_color_map = cv2.applyColorMap((255 * tensor[4]/ (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        weights_color_map = cv2.applyColorMap((255 * tensor[5] / (tensor[5].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        pil_input = np.array(pil_input)
        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
        pil_gt_matched = Image.fromarray(cv2.cvtColor(gt_matched_color_map, cv2.COLOR_BGR2RGB))
        pil_weights = Image.fromarray(cv2.cvtColor(weights_color_map, cv2.COLOR_BGR2RGB))
        imgs = [pil_input, pil_label, pil_output, pil_binar, pil_gt_matched,pil_weights]
        w_num , h_num = 3, 2
        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            x, y = int(count % w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)
            target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
            count += 1
        target.save(os.path.join(exp_path, '{}_den.jpg'.format(iter)))

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    print('=' * 50)
    print(' ' + '-' * 20 )
    content = '  ['
    for key, data in scores.items():
        if isinstance(data,str):
            content +=(' ' + key + ' %s' % data)
        else:
            content += (' ' + key + ' %.3f' % data)
    content += ']'
    print(content)
    print(' ' + '-'*20 )
    best_str = '[best]'
    for key, data in train_record.items():
        best_str += (' [' + key +' %s'% data + ']')
    print(best_str)
    print('=' * 50)

def update_model(trainer, scores, args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    train_record = trainer.train_record
    epoch = trainer.epoch
    snapshot_name = 'ep_%d_iter_%d' % (epoch, trainer.i_tb)
    for key, data in scores.items():
        snapshot_name+= ('_'+ key+'_%.3f'%data)
    for key, data in  scores.items():
        if data < train_record[key]:
            train_record['best_model_name'] = snapshot_name
            to_saved_weight = trainer.net.state_dict()
            torch.save(to_saved_weight, os.path.join(args.output_dir, snapshot_name + '.pth'))
        if data < train_record[key]:
            train_record[key] = data
    latest_state = {'train_record': train_record, 'net': trainer.net.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'epoch': trainer.epoch, 'i_tb': trainer.i_tb}
    torch.save(latest_state, os.path.join(args.output_dir, 'latest_state.pth'))
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

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, path=None, show_keypoints=False, margin=10, opencv_display=False, opencv_title='',
                            restore_transform=None, id0=None, id1=None):
    image0 = np.array(restore_transform(image0))
    image1 = np.array(restore_transform(image1))
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    H0, W0, C = image0.shape
    H1, W1, C = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255 * np.ones((H, W, C), np.uint8)
    out[:H0, :W0,:] = image0
    out[:H1, W0 + margin:, :] = image1
    out_by_point = out.copy()
    point_r_value = 15
    thickness = 3
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        for x, y in kpts0:
            cv2.circle(out, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, white, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_by_point, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 3, white, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_by_point, (x + margin + W0, y), point_r_value, blue, thickness, lineType=cv2.LINE_AA)
        if id0 is not  None:
            for i, (id, centroid) in enumerate(zip(id0, kpts0)):
                cv2.putText(out, str(id), (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if id1 is not None:
            for i, (id, centroid) in enumerate(zip(id1, kpts1)):
                cv2.putText(out, str(id), (centroid[0] + margin + W0, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out_by_point, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out_by_point, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
        cv2.imwrite(str('point_' + path), out_by_point)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)
    return out, out_by_point
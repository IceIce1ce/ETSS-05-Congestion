import os
import cv2
import torch
import numpy as np
import torch.distributed as dist

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters)**(power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    print('=' * 50)
    print(' ' + '-' * 20)
    content = '  ['
    for key, data in scores.items():
        if isinstance(data,str):
            content += (' ' + key + ' %s' % data)
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

def update_model(trainer, scores, args):
    train_record = trainer.train_record
    epoch = trainer.epoch
    snapshot_name = 'ep_%d_iter_%d' % (epoch, trainer.i_tb)
    for key, data in scores.items():
        snapshot_name += ('_' +  key + '_%.3f' % data)
    for key, data in  scores.items():
        if data < train_record[key] :
            train_record['best_model_name'] = snapshot_name
            to_saved_weight = trainer.model.state_dict()
            torch.save(to_saved_weight, os.path.join(args.output_dir, snapshot_name + '.pth'))
        if data < train_record[key]:
            train_record[key] = data
    latest_state = {'train_record': train_record, 'net': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'epoch': trainer.epoch, 'i_tb':trainer.i_tb}
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

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def save_visual_results(data, restor_transform, save_base, iter_num, rank):
    assert (len(data) - 1) % 2 == 0
    num = (len(data) - 1) // 2
    h = data[0].size(2)
    w = data[0].size(3)
    batch_size = data[0].size(0)
    margin = 5
    W = w * len(data) + margin * num + 3 * margin * num
    H = h * batch_size + margin * (batch_size - 1)
    out = np.zeros((H, W, 3))
    start_h = 0
    for i in range(batch_size):
        start_w = 0
        img = cv2.cvtColor(np.array(restor_transform(data[0][i])), cv2.COLOR_RGB2BGR)
        out[start_h:start_h + h, start_w:start_w + w] = img
        start_w += w + 3 * margin
        for j in range(num):
            data_map = data[1 + j*2][i].detach().cpu().numpy()
            vis_data_map = change2map(data_map.copy())
            out[start_h:start_h + h, start_w:start_w + w] = vis_data_map
            start_w += w + margin
            data_map = data[1 + j*2 + 1][i].detach().cpu().numpy()
            vis_data_map = change2map(data_map.copy())
            out[start_h:start_h + h, start_w:start_w + w] = vis_data_map
            start_w += w + 3 * margin
        start_h += h + margin
    if not os.path.exists(save_base):
        os.makedirs(save_base, exist_ok=True)
    cv2.imwrite(os.path.join(save_base, "{}_{}_visual.jpg".format(rank, iter_num)), out)

def save_test_visual(visual_maps, imgs, scene_name, restor_transform, save_path, iter, rank):
    visual_data = [visual_maps[:, i, :, :] for i in range(visual_maps.shape[1])]
    visual_data = [torch.stack(imgs, dim=0)] + visual_data
    save_visual_results(visual_data, restor_transform, os.path.join(save_path, scene_name), iter, rank)

def change2map(intput_map):
    intput_map = intput_map.squeeze(0)
    vis_map = (intput_map - intput_map.min()) / (intput_map.max() - intput_map.min() + 1e-5)
    vis_map = (vis_map * 255).astype(np.uint8)
    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    return vis_map

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt = torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2)
    pre_crowdflow_cnt = pre_dict['first_frame']
    gt_crowdflow_cnt = gt_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'], pre_dict['outflow'], gt_dict['inflow'], gt_dict['outflow']), 0):
        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]
        if idx % intervals == 0 or  idx == len(pre_dict['inflow']) - 1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]
    return pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE': torch.zeros(scene_cnt,2), 'WRAE': torch.zeros(scene_cnt,2), 'MIAE': torch.zeros(0), 'MOAE': torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate(zip(scenes_pred_dict, scene_gt_dict), 0):
        time = pre_dict['time']
        pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt = compute_metrics_single_scene(pre_dict, gt_dict, intervals)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i, :] = torch.tensor([mae / (gt_crowdflow_cnt + 1e-10), time])
        metrics['MIAE'] = torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:, 0] - inflow_cnt[:, 1])])
        metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])
    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1])**2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:, 0] * (metrics['WRAE'][:, 1] / (metrics['WRAE'][:, 1].sum() + 1e-10))) * 100
    MIAE = torch.mean(metrics['MIAE'])
    MOAE = torch.mean(metrics['MOAE'])
    return MAE, MSE, WRAE, MIAE, MOAE, metrics['MAE']
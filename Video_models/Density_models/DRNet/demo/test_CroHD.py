import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.getcwd())
import cv2
from model.VIC import Video_Individual_Counter
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
import os
import numpy as np
import torch
from config import cfg
from importlib import import_module
from datasets.dataset import TestDataset
from torch.utils.data import  DataLoader
import torchvision.transforms as standard_transforms
import matplotlib
matplotlib.use('Agg')
import misc.transforms as own_transforms
import matplotlib.pyplot as plt
import PIL.Image as Image

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

def createRestore(mean_std):
    return standard_transforms.Compose([own_transforms.DeNormalize(*mean_std), standard_transforms.ToPILImage()])

def test(cfg_data):
    # test loader
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*cfg_data.MEAN_STD)])
    sub_dataset = TestDataset(scene_name='HT21-02', base_path=args.input_dir, img_transform=img_transform, interval=args.test_intervals, target=True, datasetname=args.type_dataset)
    test_loader = DataLoader(sub_dataset, batch_size=cfg_data.VAL_BATCH_SIZE, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    restore_transform = createRestore(cfg_data.MEAN_STD)
    # model
    net = Video_Individual_Counter(cfg, cfg_data)
    state_dict = torch.load(args.ckpt_dir)
    net.load_state_dict(state_dict, strict=True)
    print('Load ckpt from:', args.ckpt_dir)
    net.eval()
    scenes_pred_dict = []
    if args.skip_flag:
        intervals = 1
    else:
        intervals = args.test_intervals
    for scene_id, sub_valset in enumerate([test_loader], 0):
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + args.test_intervals
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        time = []
        cnt  = []
        gt_cnt = []
        for vi, data in enumerate(gen_tqdm, 0):
            img,target = data
            img,target = img[0], target[0]
            img = torch.stack(img, 0)
            with torch.no_grad():
                b, c, h, w = img.shape
                if h % 16 != 0:
                    pad_h = 16 - h % 16
                else:
                    pad_h = 0
                if w % 16 != 0:
                    pad_w = 16 - w % 16
                else:
                    pad_w = 0
                pad_dims = (0, pad_w, 0, pad_h)
                img = F.pad(img, pad_dims, "constant")
                if vi % args.test_intervals == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'

                if frame_signal == 'match' or not args.skip_flag:
                    pred_map, gt_den, matched_results = net.val_forward(img,target)
                    pred_cnt = pred_map[0].sum().item()
                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                        gt_dict['first_frame'] = len(target[0]['person_id'])
                    pred_dict['inflow'].append(matched_results['pre_inflow'])
                    pred_dict['outflow'].append(matched_results['pre_outflow'])
                    gt_dict['inflow'].append(matched_results['gt_inflow'])
                    gt_dict['outflow'].append(matched_results['gt_outflow'])
                time.append(round(vi / intervals * 3., 2))
                if frame_signal == 'match':
                    pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, intervals=intervals)
                    cnt.append(pre_crowdflow_cnt)
                    gt_cnt.append(gt_crowdflow_cnt)
                    print('Den pred: {:.2f}, Pred crowd flow: {:.2f}, Pred inflow: {:.2f}'.format(pred_cnt, pre_crowdflow_cnt, matched_results['pre_inflow']))
                else:
                    cnt.append(cnt[-1])
                    gt_cnt.append(gt_cnt[-1])
                kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()
                matches = matched_results['matches0'].cpu().numpy()
                confidence = matched_results['matching_scores0'].cpu().numpy()
                if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                    save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(), img[1].clone(), args.test_intervals, args.output_dir, time, cnt, gt_cnt, restore_transform)
        scenes_pred_dict.append(pred_dict)

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt = torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2)
    pre_crowdflow_cnt = pre_dict['first_frame']
    gt_crowdflow_cnt =  gt_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'], pre_dict['outflow'], gt_dict['inflow'], gt_dict['outflow']), 0):
        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]
        if idx % intervals == 0 or idx== len(pre_dict['inflow']) - 1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]
    return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, path=None, show_keypoints=False, margin=10, opencv_display=False, opencv_title='', restore_transform=None):
    image0 = np.array(restore_transform(image0))
    image1 = np.array(restore_transform(image1))
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    H0, W0, C = image0.shape
    H1, W1, C = image1.shape
    pre_inflow = np.zeros((H0, W0, 3)).astype(np.uint8)
    pre_outflow = np.zeros((H1, W1, 3)).astype(np.uint8)
    H, W = max(H0, H1) + 50, W0 + 1600 + margin
    out = 255 * np.ones((H, W, C), np.uint8)
    out[:H0, :W0,:] = image1
    out_by_point = out.copy()
    point_r_value = 10
    thickness = 3
    white = (255, 255, 255)
    RoyalBlue1 = np.array([255, 118, 72])
    red = [0, 0, 255]
    green = [0, 255, 0]
    pre_inflow[:, :, 0:3] = RoyalBlue1
    pre_outflow[:, :, 0:3] = RoyalBlue1
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        for x, y in kpts1:
            cv2.circle(out, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, white, -1, lineType=cv2.LINE_AA)
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        cv2.circle(out, (x1, y1), point_r_value, green, thickness, lineType=cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
        cv2.imwrite(str('point_'+path), out_by_point)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)
    return out,out_by_point

def save_visImg(kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals, save_path, time=None, cnt=None, gt_cnt=None, restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1, 2)
    mkpts1 = kpts1[matches[valid]].reshape(-1, 2)
    color = cm.jet(confidence[valid])

    def fig2data(fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
        rgb_image = image[:, :, :3]
        return rgb_image

    fig, ax = plt.subplots(figsize = (16, 10))
    ax.cla()
    plt.tick_params(labelsize=28)
    plt.xlim(0, 130)
    plt.ylim(0, 1400)
    ax.set_xlabel('Time (s)', fontsize=32)
    ax.set_ylabel('pedestrian number', fontsize=32)
    ax.plot(time, cnt, 'b', lw=4)
    ax.plot(time, gt_cnt, 'r', lw=4)
    plt.legend(['Predicted: ' + str(np.around(cnt[-1], 2)), 'Ground Truth: '+str(np.around(gt_cnt[-1], 2))], loc='upper left', fontsize=32)
    plt.show()
    curve = fig2data(fig)
    out, out_by_point = make_matching_plot_fast(last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, path=None, show_keypoints=True, restore_transform=restore_transform)
    H0, W0, _ = out.shape
    H1, W1, _ = curve.shape
    H_s = int((H0 - H1) / 2)
    out[H_s:H_s + H1, W0 - W1:, :] = curve
    red = (0, 0, 255)
    cv2.circle(out, (20, 1080 + 27), 10, red, thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(out, 'People who come into the scene during time: [' + str(round(time[-1] - 3, 1)) + 's~' + str(round(time[-1],1)) + 's]', (40, 1080 + 42), cv2.FONT_HERSHEY_DUPLEX, 1.5, [0,0,0], 2, cv2.LINE_AA)
    cv2.circle(out, (1700, 1080 + 27), 10, [0, 255, 0], thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(out, 'People who still stay in the scene compared with time: ' + str(round(time[-1] - 3, 1)) + 's', (1740, 1080 + 42), cv2.FONT_HERSHEY_DUPLEX, 1.5, [0, 0, 0], 2, cv2.LINE_AA)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        stem = '{}_{}_matches'.format(vi, vi + intervals)
        out_file = str(Path(save_path, stem + '.jpg'))
        print('Writing image to {}'.format(out_file))
        cv2.imwrite(out_file, out)

def img_to_vid(img_path, video_path, fps, size):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    images = os.listdir(img_path)
    images.sort()
    images.sort(key=lambda x: int(x.split('_')[0]))
    vw = cv2.VideoWriter(video_path, fourcc, fps, size)
    for file in images:
        imagefile = os.path.join(img_path, file)
        try:
            new_frame = cv2.imread(imagefile)
            new_frame=cv2.resize(new_frame, size, interpolation=cv2.INTER_AREA)
            vw.write(new_frame)
        except Exception as exc:
            print(imagefile, exc)
    vw.release()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--input_dir', type=str, default='data/HT21/train')
    parser.add_argument('--output_dir', type=str, default='saved_ht21_demo')
    parser.add_argument('--test_intervals', type=int, default=75)
    parser.add_argument('--skip_flag', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--ckpt_dir', type=str, default='saved_ht21/HT21.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    setup_seed(args.seed)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    test(datasetting.cfg_data)
    img_to_vid(args.output_dir, os.path.join(args.output_dir, 'demo.mp4'), 25, (int((1920 + 1600) / 3), int((1130) / 3)))
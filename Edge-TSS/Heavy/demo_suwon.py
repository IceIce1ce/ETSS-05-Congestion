import warnings
warnings.filterwarnings('ignore')
import os
from collections import OrderedDict
from config import return_args, args
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from utils import setup_seed
import nni
from nni.utils import merge_parameter
import util.misc as utils
import torch
import numpy as np
import cv2
import torch.nn as nn
from Networks.CDETR import build_model
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()
history_counts = []

def generate_chart_image(history_counts, total_frames, chart_width, chart_height):
    fig, ax = plt.subplots(figsize=(chart_width / 100, chart_height / 100), dpi=100)
    canvas = FigureCanvas(fig)
    max_count = max(max(history_counts), 1) if history_counts else 1
    step_v = max_count / 5
    v_ticks = [int(step_v * i) for i in range(6)]
    step_h = total_frames / 5
    h_ticks = [int(step_h * i) for i in range(6)]
    ax.axhspan(0, 24, facecolor="green", alpha=0.15, label="Low Congestion (0–24)")
    ax.axhspan(25, 100, facecolor="yellow", alpha=0.15, label="Medium Congestion (25–100)")
    ax.axhspan(101, max_count, facecolor="red", alpha=0.15, label="High Congestion (>100)")
    ax.plot(range(len(history_counts)), history_counts, color='black', linewidth=2)
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, max_count)
    ax.set_xticks(h_ticks)
    ax.set_yticks(v_ticks)
    ax.set_xlabel("Frame", fontsize=30)
    ax.set_ylabel("Count", fontsize=30)
    ax.set_xticks(h_ticks)
    ax.set_yticks(v_ticks)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.legend(loc='upper right', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    canvas.draw()
    chart_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    chart_img = chart_img.reshape(canvas.get_width_height()[::-1] + (3,))
    chart_img = cv2.resize(chart_img, (chart_width, chart_height))
    plt.close(fig)
    return chart_img

def main(args):
    utils.init_distributed_mode(return_args)
    # model
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    # resume training
    if args['pre']:
        if os.path.isfile(args['pre']):
            checkpoint = torch.load(args['pre'])['state_dict']
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('bbox', 'point')
                new_state_dict[name] = v
            print("Load ckpt from: {}".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(new_state_dict)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("No ckpt found")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoCapture(args['video_path'])
    width = 1024
    height = 768
    out = cv2.VideoWriter('output.mp4', fourcc, 5, (width * 2, height * 2))
    # calculate FPS
    start_time = time.time()
    frame_count = 0
    while True:
        try:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (width, height))
        except:
            cap.release()
            break
        frame = frame.copy()
        image = tensor_transform(frame)
        image = img_transform(image)
        width, height = image.shape[2], image.shape[1]
        num_w = int(width / 256)
        num_h = int(height / 256)
        image = image.view(3, num_h, 256, width).view(3, num_h, 256, num_w, 256)
        image = image.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, 256, 256).permute(1, 0, 2, 3)
        with torch.no_grad():
            image = image.cuda()
            outputs = model(image)
            out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), args['num_queries'], dim=1)
            topk_points = topk_indexes // out_logits.shape[2]
            out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
            out_point = out_point * 256
            value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)
            crop_size = 256
            kpoint_map, density_map, frame_overlay, count = show_map(value_points, frame, width, height, crop_size, num_h, num_w)
            history_counts.append(count)
            chart_height = height
            chart_img = generate_chart_image(history_counts, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), width * 2, chart_height)
            top_row = np.hstack((frame_overlay, density_map))
            res = np.vstack((top_row, chart_img))
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(res, "Count:" + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            out.write(res)
            print(f"Frame: {frame_count}, Count: {count}, FPS: {fps:.2f}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def show_map(out_pointes, frame, width, height, crop_size, num_h, num_w):
    kpoint_list = []
    confidence_list = []
    for i in range(len(out_pointes)):
        out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy() # [500]
        out_point = out_pointes[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
        k = np.zeros((crop_size, crop_size)) # [256, 256]
        c_map = np.zeros((crop_size, crop_size)) # [256, 256]
        for j in range(len(out_point)):
            if out_value[j] < 0.25:
                break
            x = int(out_point[j][0])
            y = int(out_point[j][1])
            k[x, y] = 1
        kpoint_list.append(k)
        confidence_list.append(c_map)
    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0) # [1, 12, 256, 256]
    kpoint = kpoint.view(num_h, num_w, crop_size, crop_size).permute(0, 2, 1, 3).contiguous().view(num_h, crop_size, width).view(height, width).cpu().numpy() # [768, 1024]
    density_map = gaussian_filter(kpoint.copy(), 6) # [768, 1024]
    density_map = density_map / np.max(density_map) * 255 # [768, 1024]
    density_map = density_map.astype(np.uint8) # [768, 1024]
    density_map = cv2.applyColorMap(density_map, 2) # [768, 1024, 3]
    pred_coor = np.nonzero(kpoint)
    count = len(pred_coor[0])
    point_map = np.zeros((int(kpoint.shape[0]), int(kpoint.shape[1] ), 3), dtype="uint8") + 255 # [768, 1024, 3]
    for i in range(count):
        w = int(pred_coor[1][i])
        h = int(pred_coor[0][i])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)
        cv2.circle(frame, (w, h), 3, (0, 255, 50), -1)
    return point_map, density_map, frame, count

if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    setup_seed(args.seed)
    main(params)
import warnings
warnings.filterwarnings('ignore')
from Networks.HR_Net.seg_hrnet import get_seg_model
import torch.nn as nn
from torchvision import transforms
import scipy
import cv2
import numpy as np
import torch
import os
import nni
from nni.utils import merge_parameter
from config import return_args

def main(args):
    # data transform
    img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_transform = transforms.ToTensor()
    # model
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    if args['pre']:
        if os.path.isfile(args['pre']):
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
            print('Load ckpt from:', args['pre'])
        else:
            print('No ckpt found at:', args['pre'])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(args['video_path'])
    ret, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]
    # out = cv2.VideoWriter('demo.avi', fourcc, 30, (width, height))
    out_w, out_h = width * 2, height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo.avi', fourcc, 30, (out_w, out_h))
    while True:
        try:
            ret, frame = cap.read()
            # scale_factor = 0.5
            # frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            # ori_img = frame.copy()
        except:
            cap.release()
            break
        frame = frame.copy()
        image = tensor_transform(frame)
        image = img_transform(image).unsqueeze(0)
        with torch.no_grad():
            d6 = model(image)
            count, pred_kpoint = counting(d6)
            point_map = generate_point_map(pred_kpoint)
            box_img = generate_bounding_boxes(pred_kpoint, frame)
            show_fidt = show_fidt_func(d6.data.cpu().numpy())
            # res1 = np.hstack((ori_img, show_fidt))
            # res2 = np.hstack((box_img, point_map))
            # res = np.vstack((res1, res2))
            res = np.hstack((box_img, show_fidt))
            cv2.putText(res, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(res)
        print("Pred: %.2f" % count)

def counting(input):
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input
    input[input < 100.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1
    if input_max < 0.1:
        input = input * 0
    count = int(torch.sum(input).item())
    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()
    return count, kpoint

def generate_point_map(kpoint):
    rate = 1
    pred_coor = np.nonzero(kpoint)
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)
    return point_map

def generate_bounding_boxes(kpoint, Img_data):
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    if pts.shape[0] > 0:
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        distances, locations = tree.query(pts, k=4)
        for index, pt in enumerate(pts):
            pt2d = np.zeros(kpoint.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if np.sum(kpoint) > 1:
                sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
            else:
                sigma = np.average(np.array(kpoint.shape)) / 2. / 2. # 1 point
            sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.04)
            if sigma < 6:
                t = 2
            else:
                t = 2
            Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)), (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)
    return Img_data

def show_fidt_func(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1

if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    main(params)
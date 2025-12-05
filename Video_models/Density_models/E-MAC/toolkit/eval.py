import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.io import loadmat
import os
import re
import argparse

def get_seq_class(seq, set, args):
    backlight = ["DJI_0021", "DJI_0022", "DJI_0032", "DJI_0202", "DJI_0339", "DJI_0340", "DJI_0463", "DJI_0003"]
    fly = ["DJI_0177", "DJI_0174", "DJI_0022", "DJI_0180", "DJI_0181", "DJI_0200", "DJI_0544", "DJI_0012", "DJI_0178", "DJI_0343", "DJI_0185", "DJI_0195", "DJI_0996", "DJI_0977",
           "DJI_0945", "DJI_0946", "DJI_0091", "DJI_0442", "DJI_0466", "DJI_0459", "DJI_0464"]
    angle_90 = ["DJI_0179", "DJI_0186", "DJI_0189", "DJI_0191", "DJI_0196", "DJI_0190", "DJI_0070", "DJI_0091"]
    mid_size = ["DJI_0012", "DJI_0013", "DJI_0014", "DJI_0021", "DJI_0022", "DJI_0026", "DJI_0028", "DJI_0028", "DJI_0030", "DJI_0028", "DJI_0030", "DJI_0034", "DJI_0200", "DJI_0544",
                "DJI_0463", "DJI_0001", "DJI_0149"]
    light = "sunny"
    bird = "stand"
    angle = "60"
    size = "small"
    if seq in backlight:
        light = "backlight"
    if seq in fly:
        bird = "fly"
    if seq in angle_90:
        angle = "90"
    if seq in mid_size:
        size = "mid"
    count = "sparse"
    loca = loadmat(os.path.join(args.input_dir, set, "ground_truth", "GT_img" + str(seq[-3:]) + "000.mat"))["locations"]
    if loca.shape[0] > 150:
        count = "crowded"
    return [light, angle, bird, size, count]

def main(args):
    with open(args.output_dir, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        match = re.match(r"(\d+): err: ([\d.]+), gt_count: ([\d.]+), pred_count: ([\d.]+), name: (.+)", line)
        if match:
            error = float(match.group(2))
            gt_count = float(match.group(3))
            pred_count = float(match.group(4))
            name = match.group(5)
            data.append({"error": error, "gt_count": gt_count, "pred_count": pred_count, "name": name})
    preds = []
    gts = []
    preds_hist = [[] for i in range(10)]
    gts_hist = [[] for i in range(10)]
    attri = ["sunny", "backlight", "crowded", "sparse", "60", "90", "stand", "fly", "small", "mid"]
    for d in data:
        name = d["name"]
        gt_count = d["gt_count"]
        pred_count = d["pred_count"]
        seq = "DJI_" + str(int(name[3:6])).zfill(4)
        cur_attris = get_seq_class(seq, "test", args)
        preds.append(pred_count)
        gts.append(gt_count)
        for cur_attri in cur_attris:
            preds_hist[attri.index(cur_attri)].append(pred_count)
            gts_hist[attri.index(cur_attri)].append(gt_count)
    test_log = "[Test]: MAE: {:.2f}, MSE: {:.2f}\n".format(mean_absolute_error(gts, preds), mean_squared_error(gts, preds))
    for i in range(10):
        if len(preds_hist[i]) == 0:
            continue
        test_log_attri = "[{}]: MAE: {:.2f}, MSE: {:.2f}\n".format(attri[i], mean_absolute_error(gts_hist[i], preds_hist[i]), mean_squared_error(gts_hist[i], preds_hist[i]))
        test_log += test_log_attri
    print(test_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='DroneBird')
    parser.add_argument('--input_dir', type=str, default='datasets/DroneBird')
    parser.add_argument('--output_dir', type=str, default='saved_dronebird/DroneBird_result.txt')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)
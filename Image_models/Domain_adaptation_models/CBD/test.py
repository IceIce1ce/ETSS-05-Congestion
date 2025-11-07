import warnings
warnings.filterwarnings("ignore")
import math
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import argparse
from models.OC import ObjectCounter

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

def count_from_fidt(input):
    input = torch.Tensor(input)
    input = input.reshape(-1, 1, input.shape[-2], input.shape[-1])
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1
    if input_max < 0.1:
        input = input * 0
    count = torch.sum(input).item()
    return count

def list_image_files(img_dir: Path) -> List[str]:
    files = [f.name for f in sorted(img_dir.iterdir()) if f.is_file() and not f.name.startswith(".")]
    return files

def load_model(model_path: Path) -> ObjectCounter:
    net = ObjectCounter([0])
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    print('Load ckpt from:', model_path)
    net.cuda()
    net.eval()
    return net

def evaluate(file_list: List[str], model: ObjectCounter) -> None:
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()
    for filename in file_list:
        img_path = IMG_DIR / filename
        stem = img_path.stem
        den_path = DEN_DIR / f"{stem}.npy"
        den = np.load(den_path)
        gt_count = count_from_fidt(den)
        img = Image.open(img_path)
        if img.mode == "L":
            img = img.convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            pred_map = model.test_forward(img_tensor)
        pred_count = count_from_fidt(pred_map)
        err = gt_count - pred_count
        mae_meter.update(abs(err))
        mse_meter.update(err * err)
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae_meter.avg, math.sqrt(mse_meter.avg)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='PUCPR')
    parser.add_argument('--input_dir', type=str, default='datasets/PUCPR/test')
    parser.add_argument('--ckpt_dir', type=str, default='saved_pucpr/CA2PU_parameters.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    MEAN_STD = {"PUCPR": ([0.52323937416, 0.52659797668, 0.48122045398], [0.21484816074, 0.20566709340, 0.22544583678]),
                "CARPK": ([0.46704635024, 0.49598187208, 0.47164431214], [0.24702641368, 0.23411691189, 0.24729225040])}
    IMG_DIR = Path(args.input_dir) / "img"
    DEN_DIR = Path(args.input_dir) / "den"
    torch.backends.cudnn.benchmark = True
    img_transform = T.Compose([T.ToTensor(), T.Normalize(*MEAN_STD[args.type_dataset])])
    file_list = list_image_files(IMG_DIR)
    model = load_model(args.ckpt_dir)
    evaluate(file_list, model)
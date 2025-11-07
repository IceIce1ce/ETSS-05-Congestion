import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from models.OC import ObjectCounter
import os

def count_from_fidt(input): # [1, 1, 720, 1280]
    input = torch.Tensor(input)
    input = input.reshape(-1,1,input.shape[-2],input.shape[-1])
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input # [1, 1, 720, 1280]
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1
    if input_max < 0.1:
        input = input * 0
    count = torch.sum(input).item()
    return count

def load_model(model_path: Path) -> ObjectCounter:
    net = ObjectCounter([0])
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    net.load_state_dict(state)
    print('Load ckpt from:', model_path)
    net.cuda()
    net.eval()
    return net

def _to_numpy_hw(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = x.astype(np.float32, copy=False) # [1, 1, 720, 1280]
    while x.ndim > 2:
        x = x[0]
    return x

def _resize_bilinear_hw(arr_hw: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if arr_hw.shape == (h, w):
        return arr_hw
    ten = torch.from_numpy(arr_hw)[None, None, :, :]
    out = F.interpolate(ten, size=(h, w), mode="bilinear", align_corners=False)
    return out[0, 0].cpu().numpy()

def _normalize_for_vis(dmap: np.ndarray) -> np.ndarray:
    if np.any(dmap > 0):
        vmax = float(np.percentile(dmap, 99.5))
        if vmax <= 1e-12:
            vmax = float(dmap.max() + 1e-6)
    else:
        vmax = float(dmap.max() + 1e-6)
    vmax = 1.0 if vmax <= 1e-12 else vmax
    d_clipped = np.clip(dmap, 0.0, vmax)
    return d_clipped / (vmax + 1e-12)

def predict_and_save(input_dir, output_dir, args):
    # model
    model = load_model(args.ckpt_dir)
    img = Image.open(input_dir)
    if img.mode == "L":
        img = img.convert("RGB")
    img_tensor = img_transform(img).unsqueeze(0).cuda() # [1, 3, 720, 1280]
    with torch.no_grad():
        pred_map = model.test_forward(img_tensor) # [1, 3, 720, 1280]
    pred_count = float(count_from_fidt(pred_map))
    pred_np = _to_numpy_hw(pred_map) # [720, 1280]
    if args.upsample_to_image:
        H, W = img.size[1], img.size[0]
        pred_np = _resize_bilinear_hw(pred_np, (H, W))
    pred_vis = _normalize_for_vis(pred_np) # [720, 1280]
    plt.imsave(output_dir, pred_vis, cmap='jet')
    return pred_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='PUCPR')
    parser.add_argument('--input_dir', type=str, default='images/PUCPR_vis3.png')
    parser.add_argument('--output_dir', type=str, default='vis_pucpr')
    parser.add_argument('--ckpt_dir', type=str, default='saved_pucpr/CA2PU_parameters.pth')
    parser.add_argument('--upsample_to_image', type=bool, default=True)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    MEAN_STD = {"PUCPR": ([0.52323937416, 0.52659797668, 0.48122045398], [0.21484816074, 0.20566709340, 0.22544583678]),
                "CARPK": ([0.46704635024, 0.49598187208, 0.47164431214], [0.24702641368, 0.23411691189, 0.24729225040])}
    torch.backends.cudnn.benchmark = True
    img_transform = T.Compose([T.ToTensor(), T.Normalize(*MEAN_STD[args.type_dataset])])
    count = predict_and_save(args.input_dir, os.path.join(args.output_dir, args.input_dir.split('/')[-1]), args)
    print(f"Predicted count: {count:.2f}")
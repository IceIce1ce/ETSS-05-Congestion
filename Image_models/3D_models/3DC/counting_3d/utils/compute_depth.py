import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from PIL import Image
import argparse
import torch
import sys
import os
sys.path.append(os.path.abspath("."))
from ext.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

def make_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_configs = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}, "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                     "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}, "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}}
    encoder = "vitl" # 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f"weights/depth_anything_v2_{encoder}.pth", map_location="cpu"))
    model = model.to(DEVICE).eval()
    return model

def compute_depth(model, img):
    depth = model.infer_image(img)
    return depth

if __name__ == "__main__":
    model = make_model()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", default="", type=str)
    parser.add_argument("--image-output", default="", type=str)
    parser.add_argument("--dataset", default="", type=str)
    args = parser.parse_args()

    if args.image_input != "" and args.image_output != "":
        print("Computing depth from: {} to: {}".format(args.image_input, args.image_output))
        if os.path.exists(args.image_input):
            raw_img = Image.open(args.image_input)
            raw_img = np.array(raw_img)
            depth_uint8 = compute_depth(model, raw_img)
            cv2.imwrite(args.image_output, depth_uint8)
            print("Saved to:", args.image_output)
        else:
            raise ValueError("Input image doesn't exist")
    elif args.dataset != "":
        for subfolder in sorted(os.listdir(args.dataset)):
            subfolder_path = os.path.join(args.dataset, subfolder)
            if os.path.isdir(subfolder_path) and subfolder.endswith("_0"):
                image_path = os.path.join(subfolder_path, "MultiView/RGB/RGB0000.png")
                output_path = os.path.join(subfolder_path, "nadir_depth.png")
                if os.path.exists(image_path):
                    raw_img = Image.open(image_path)
                    raw_img = np.array(raw_img)
                    depth_uint8 = compute_depth(model, raw_img)
                    cv2.imwrite(output_path, depth_uint8)
                    print("Saved to:", output_path)
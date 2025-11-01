import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from gamma_model import DinoRegression
from utils.append_to_json import append_to_json
from find_nadir_image import find_nadir_image
from utils.compute_depth import compute_depth, make_model
import os

def main(args):
    image_size = 14 * args.encoded_image_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = find_nadir_image(args.input_folder)
    depth_model = make_model()
    depth_uint8 = compute_depth(depth_model, img)
    depth_map = depth_uint8
    if args.output_folder != "":
        plt.imsave(args.output_folder + "/estimated_depth.png", depth_map)
        plt.imsave(args.output_folder + "/nadir.png", img)
    feats_dim = 768 if args.dino_arch == "dinov2_vitb14" else 1024
    config = {"device": device, "MODEL_CONFIG": {"feats_dim": feats_dim, "encoded_image_size": args.encoded_image_size}}
    model = DinoRegression(args.dino_arch, config)
    print(f"LOADING density_net_{args.exp_name}.pth")
    model_path = os.path.join("weights", f"density_net_{args.exp_name}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    depth_tensor = torch.tensor(depth_map).unsqueeze(0).unsqueeze(0).half().to(device)

    def crop_tensor(tensor, crop_fraction=0.3):
        _, _, h, w = tensor.shape
        crop_h = int(h * crop_fraction)
        crop_w = int(w * crop_fraction)
        cropped_tensor = tensor[:, :, crop_h : h - crop_h, crop_w : w - crop_w]
        return cropped_tensor

    depth_tensor = crop_tensor(depth_tensor)
    depth_resized = F.interpolate(depth_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    depth_resized_3ch = depth_resized.repeat(1, 3, 1, 1)
    depth_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    depth = depth_resized_3ch
    depth /= depth.max()
    print("Range of depth:", depth.min().item(), "to", depth.max().item())
    assert depth.shape == (1, 3, image_size, image_size), f"got shape {depth.shape}"
    normalized_depth_tensor = depth_normalize(depth[0]).unsqueeze(0)
    if args.output_folder != "":
        plt.imsave(args.output_folder + "/estimated_depth_cropped.png", depth[0, 0].cpu().numpy())
    out = model(normalized_depth_tensor.float())
    print("Predicted density:", out.item())
    if args.output_folder != "":
        append_to_json(args.output_folder + "/results.json", "volume_usage_" + args.exp_name, out.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="1_0", type=str)
    parser.add_argument("--input-folder", default="", type=str)
    parser.add_argument("--output-folder", default="", type=str)
    parser.add_argument('--dino_arch', type=str, default='dinov2_vitb14')
    parser.add_argument('--encoded_image_size', type=int, default=32)
    args = parser.parse_args()
    main(args)
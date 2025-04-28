import torch
from argparse import ArgumentParser
import os, json
from tqdm import tqdm
current_dir = os.path.abspath(os.path.dirname(__file__))
from datasets import NWPUTest, Resize2Multiple
from models import get_model
from utils import get_config, sliding_window_predict

def main(args: ArgumentParser):
    _ = get_config(vars(args).copy(), mute=False)
    if args.regression:
        bins, anchor_points = None, None
    else:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)]["nwpu"]
        bins = config["bins"][args.granularity]
        anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"][args.granularity]["middle"]
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]
    args.bins = bins
    args.anchor_points = anchor_points
    # model
    model = get_model(backbone=args.model, input_size=args.input_size, reduction=args.reduction, bins=bins, anchor_points=anchor_points, prompt_type=args.prompt_type,
                      num_vpt=args.num_vpt, vpt_drop=args.vpt_drop, deep_vpt=not args.shallow_vpt)
    state_dict = torch.load(args.ckpt_dir, map_location="cpu")
    state_dict = state_dict if "best" in os.path.basename(args.ckpt_dir) else state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()
    model.eval()
    sliding_window = args.sliding_window
    if args.sliding_window:
        window_size = args.input_size
        stride = window_size // 2 if args.stride is None else args.stride
        if args.resize_to_multiple:
            transforms = Resize2Multiple(base=args.input_size)
        else:
            transforms = None
    else:
        window_size, stride = None, None
        transforms = None
    # test loader
    dataset = NWPUTest(transforms=transforms, return_filename=True)
    image_ids = []
    preds = []
    for idx in tqdm(range(len(dataset)), desc="Testing on NWPU"):
        image, image_path = dataset[idx]
        image = image.unsqueeze(0) # [1, 3, 1536, 2304]
        image = image.cuda()
        with torch.set_grad_enabled(False):
            if sliding_window:
                pred_density = sliding_window_predict(model, image, window_size, stride)
            else:
                pred_density = model(image) # [1, 1, 192, 288]
            pred_count = pred_density.sum(dim=(1, 2, 3)).item() # 232
        image_ids.append(os.path.basename(image_path).split(".")[0])
        preds.append(pred_count)
    result_dir = os.path.join(current_dir, args.output_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    weights_dir, weights_name = os.path.split(args.ckpt_dir)
    model_name = os.path.split(weights_dir)[-1]
    result_path = os.path.join(result_dir, f"{model_name}_{weights_name.split('.')[0]}.txt")
    with open(result_path, "w") as f:
        for idx, (image_id, pred) in enumerate(zip(image_ids, preds)):
            if idx != len(image_ids) - 1:
                f.write(f"{image_id} {pred}\n")
            else:
                f.write(f"{image_id} {pred}")

if __name__ == "__main__":
    parser = ArgumentParser()
    # model config
    parser.add_argument("--model", type=str, default="vgg19_ae")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32])
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--truncation", type=int, default=None)
    parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"])
    parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"])
    parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"])
    parser.add_argument("--num_vpt", type=int, default=32)
    parser.add_argument("--vpt_drop", type=float, default=0.0)
    parser.add_argument("--shallow_vpt", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    # testing config
    parser.add_argument("--sliding_window", action="store_true")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--resize_to_multiple", action="store_true")
    parser.add_argument("--zero_pad_to_multiple", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--type_dataset", type=str, default='nwpu', choices=['nwpu', 'sha', 'qnrf'])
    parser.add_argument('--output_dir', type=str, default='saved_nwpu')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    args.model = args.model.lower()
    if args.regression:
        args.truncation = None
        args.anchor_points = None
        args.bins = None
        args.prompt_type = None
        args.granularity = None
    if "clip_vit" not in args.model:
        args.num_vpt = None
        args.vpt_drop = None
        args.shallow_vpt = None
    if "clip" not in args.model:
        args.prompt_type = None
    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
        assert not (args.zero_pad_to_multiple and args.resize_to_multiple), "Cannot use both zero pad and resize to multiple."
    else:
        args.window_size = None
        args.stride = None
        args.zero_pad_to_multiple = False
        args.resize_to_multiple = False
    main(args)
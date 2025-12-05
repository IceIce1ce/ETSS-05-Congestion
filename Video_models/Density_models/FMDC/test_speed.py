import warnings
warnings.filterwarnings("ignore")
import torch
from config import cfg
from importlib import import_module
from model.video_crowd_count import video_crowd_count
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--ckpt_dir', type=str, default='saved_den_ht21/ep_1_iter_1000_mae_30.234_mse_30.726_seq_MAE_220.805_WRAE_253.799_MIAE_21.129_MOAE_22.114.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    model = video_crowd_count(cfg, datasetting.cfg_data).cuda()
    state_dict = torch.load(args.ckpt_dir)
    try:
        model.load_state_dict(state_dict["net"], strict=True)
    except:
        model.load_state_dict(state_dict, strict=True)
    print('Load ckpt from:', args.ckpt_dir)
    model.eval()
    img = torch.rand(4, 3, 768 // 2, 1024 // 2).cuda()
    start = time.time()
    den, mask, pre_outflow_map, pre_inflow_map = model.test_or_validate(img, None) # [4, 1, 384, 512], [4, 1, 384, 512]
    end = time.time() - start
    print('Latency:', end)
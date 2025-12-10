import argparse

parser = argparse.ArgumentParser()
# general config
parser.add_argument('--type_dataset', type=str, default='ShanghaiTech')
parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech')
parser.add_argument('--output_dir', type=str, default='save_sha')
parser.add_argument('--loc_dir', type=str, default='local_eval/point_files')
parser.add_argument('--write_loc', type=bool, default=False)
# training config
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--print_freq', type=int, default=200)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--pre', type=str, default=None) # pretrained model dir
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--best_pred', type=int, default=1e5)
parser.add_argument('--lr', type=float, default= 1e-4)
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4)
parser.add_argument('--preload_data', type=bool, default=False)
# testing config
parser.add_argument('--visual', type=bool, default=False) # for bounding box
parser.add_argument('--video_path', type=str, default=None) # for video demo
args = parser.parse_args()
return_args = parser.parse_args()
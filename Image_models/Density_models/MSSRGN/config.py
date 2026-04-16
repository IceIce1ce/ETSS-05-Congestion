import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type_dataset', type=str, default='Crowd-SR')
parser.add_argument('--pre', type=str, default='saved_srcrowd/model_best_x2.pth')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--upscale', type=str, default='x2')
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
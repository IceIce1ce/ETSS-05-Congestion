import argparse
import os
import numpy as np
import torch
from datasets.awcc_dataset import Crowd
from models.CC import CrowdCounter
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def eval(model, dataloader):
    model.eval()
    epoch_res = []
    for inputs, count in tqdm(dataloader):
        inputs = inputs.cuda() # [1, 3, 955, 1300]
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        outputs = model.test_forward(inputs) # [1, 1, 59, 81]
        err = count[0].item() - torch.sum(outputs).item()
        epoch_res.append(err)
    epoch_res = np.array(epoch_res) # [1600]
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    print('MSE: {:.4f}, MAE: {:.4f}'.format(mse, mae))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='datasets/jhu', type=str)
    parser.add_argument('--ckpt_dir', default='checkpoints/best.pth', type=str)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--downsample_ratio', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    print('Testing dataset:', args.input_dir.split('/')[-1])
    # test loader
    dataset = Crowd(os.path.join(args.input_dir, 'test'), args.crop_size, args.downsample_ratio, 'val')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # model
    model = CrowdCounter(args)
    ckpt = torch.load(args.ckpt_dir)['state_dict']
    msg = model.load_state_dict(ckpt)
    model.cuda()
    # test
    eval(model, dataloader)
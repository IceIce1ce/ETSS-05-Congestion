import warnings
warnings.filterwarnings("ignore")
import math
import torch
import torch.nn as nn
from torchvision import transforms
import dataset
from Networks.MSSRM import MSSRM
import numpy as np
from config import args
import os

def validate(Pre_data, model, args):
    # test loader
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])]), train=False), batch_size=args.batch_size)
    model.eval()
    mae = 0
    mse = 0
    for i, (img, target, kpoint, fname) in enumerate(test_loader):
        img = img.cuda()
        target = target.type(torch.FloatTensor).cuda()
        out1 = model(img, target, None, phase='test') # [1, 1, 341, 512]
        count = torch.sum(out1).item()
        gt_count = torch.sum(kpoint).item()
        if i % 50 == 0:
            print(fname[0], 'gt', torch.sum(kpoint).item(), "pred", int(count))
        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)
    mae = mae / len(test_loader)
    mse = math.sqrt(mse/len(test_loader))
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))

def main():
    if args.type_dataset == 'Crowd-SR':
        test_file = 'npydata/crowdsr_test.npy'
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    # model
    model = MSSRM(upscale=args.upscale).cuda()
    model = nn.DataParallel(model, device_ids=[0])
    if args.pre:
        if os.path.isfile(args.pre):
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['state_dict'])
            print('Load ckpt from:', args.pre)
        else:
            print('No ckpt found')
    validate(val_list, model, args)

if __name__ == '__main__':
    torch.cuda.manual_seed(args.seed)
    print('Testing dataset:', args.type_dataset)
    main()
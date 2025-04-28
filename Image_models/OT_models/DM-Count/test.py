import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19
import cv2
import warnings
warnings.filterwarnings("ignore")

def main(args):
    if args.type_dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(args.dataset_dir, 'test'), args.crop_size, 8, method='val')
    elif args.type_dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(args.dataset_dir, 'val'), args.crop_size, 8, method='val')
    elif args.type_dataset.lower() == 'sha' or args.type_dataset.lower() == 'shb':
        dataset = crowd.Crowd_sh(os.path.join(args.dataset_dir, 'test_data'), args.crop_size, 8, method='val')
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    # test loader
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # model
    model = vgg19()
    model.cuda()
    model.load_state_dict(torch.load(args.ckpt_dir, map_location='cuda'))
    model.eval()
    image_errs = []
    for inputs, count, name in dataloader:
        inputs = inputs.cuda() # [1, 3, 1264, 1920]
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs) # [1, 1, 158, 240]
        img_err = count[0].item() - torch.sum(outputs).item()
        print('Name: {}, Error: {:.4f}, GT: {}, Pred: {:.4f}'.format(name, img_err, count[0].item(), torch.sum(outputs).item()))
        image_errs.append(img_err)
        # visualize density map
        if args.output_dir is not None:
            vis_img = outputs[0, 0].cpu().numpy()
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.output_dir, str(name[0]) + '.png'), vis_img)
    image_errs = np.array(image_errs) # [500]
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/model_nwpu.pth')
    parser.add_argument('--dataset_dir', type=str, default='data/nwpu')
    parser.add_argument('--type_dataset', type=str, default='nwpu', choices=['nwpu', 'qnrf', 'sha', 'shb'])
    parser.add_argument('--output_dir', type=str, default='saved_density_nwpu')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)

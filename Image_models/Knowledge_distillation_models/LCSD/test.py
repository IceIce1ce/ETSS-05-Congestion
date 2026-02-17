import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import os
import numpy as np
from models.counter.models import vgg19
from datasets.crowd import Crowd
from torch.utils.data.dataloader import default_collate
from utils.utils import gen_pseudo_point, eval_loc_F1_point
from utils.pytorch_utils import AverageCategoryMeter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import torch.nn.functional as F
import cv2

def main(args):
    # test loader
    dataset_path = os.path.join(args.input_dir, args.type_dataset, args.scene, args.scene_dataset)
    dataset = Crowd(dataset_path, 'test', 8)
    dataloader = torch.utils.data.DataLoader(dataset,args.batch_size, shuffle=False, collate_fn=default_collate, num_workers=3, pin_memory=False)
    if args.pred_density_map_path:
        if not os.path.exists(args.pred_density_map_path):
            os.makedirs(args.pred_density_map_path)
    # model
    model = vgg19(args)
    model.cuda()
    model.load_state_dict(torch.load(args.ckpt_dir, map_location='cuda'))
    model.eval()
    image_errs = []
    max_dist_thresh = 100
    loc_100_metrics = {'tp_100': AverageCategoryMeter(max_dist_thresh), 'fp_100': AverageCategoryMeter(max_dist_thresh), 'fn_100': AverageCategoryMeter(max_dist_thresh)}
    for img, inputs, points, name in dataloader:
        inputs = inputs.cuda() # [1, 3, 480, 640]
        count = len(points[0])
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        inputs = inputs.repeat(4, 1, 1, 1) # [4, 3, 480, 640]
        with torch.set_grad_enabled(False):
            multi_output, _ = model(inputs) # [4, 1, 60, 80]
            output = torch.mean(multi_output.squeeze(1), 0) # [60, 80]
        pseudo_point = gen_pseudo_point(output) # [8, 2]
        img_err = count - torch.sum(output).item()
        print('Name: {}, Error: {}, GT: {:.2f}, Pred: {:.2f}'.format(name[0], img_err, count, len(pseudo_point)))
        image_errs.append(img_err)
        tp_100, fp_100, fn_100 = eval_loc_F1_point(pseudo_point.cpu().numpy(), points[0].cpu().numpy(), max_dist_thresh = max_dist_thresh)
        loc_100_metrics['tp_100'].update(tp_100)
        loc_100_metrics['fp_100'].update(fp_100)
        loc_100_metrics['fn_100'].update(fn_100)
        if args.pred_density_map_path:
            vis_img = F.upsample_bilinear(output.unsqueeze(0).unsqueeze(0), scale_factor=8)[0, 0].cpu().numpy()
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET) # [480, 640, 3]
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)
            img = img.squeeze(0)
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for x, y in pseudo_point:
                point = patches.Circle((x.item(), y.item()), 2, linewidth=2, facecolor="red")
                ax.add_patch(point)
            ax.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.tight_layout()
            output_path = os.path.join(args.pred_density_map_path, str(name[0]) + '_show.png')
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0, dpi=400)
            plt.close()
    image_errs = np.array(image_errs) # [1200]
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))
    pre_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum  + loc_100_metrics['fp_100'].sum + 1e-20)
    rec_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum  + loc_100_metrics['fn_100'].sum + 1e-20)
    f1_100 = 2 * (pre_100 * rec_100) / (pre_100 + rec_100 + 1e-20)
    print('Avg precision: {:.2f}, Avg recall: {:.2f}, Avg F1: {:.2f}'.format(pre_100.mean(), rec_100.mean(), f1_100.mean()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', default='Mall', type=str)
    parser.add_argument('--scene', default='scene_001', type=str)
    parser.add_argument('--input_dir', type=str, default='data')
    parser.add_argument('--scene_dataset', default='mall_800_1200', type=str)
    parser.add_argument('--ckpt_dir', type=str, default='saved_mall/models/counter_model_1.pth')
    parser.add_argument('--pred_density_map_path', type=str, default='saved_pred_mall')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)
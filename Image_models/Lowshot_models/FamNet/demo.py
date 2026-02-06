import warnings
warnings.filterwarnings("ignore")
import cv2
from model import CountRegressor, Resnet50FPN
from utils import visualize_output_and_save, select_exemplar_rois, MAPS, Scales, Transform, extract_features
from PIL import Image
import os
import torch
import argparse
import torch.optim as optim
from utils import MincountLoss, PerturbationLoss
from tqdm import tqdm

def main(args):
    # model
    resnet50_conv = Resnet50FPN()
    regressor = CountRegressor(6, pool='mean')
    resnet50_conv.cuda()
    regressor.cuda()
    regressor.load_state_dict(torch.load(args.ckpt_dir))
    print('Load ckpt from:', args.ckpt_dir)
    resnet50_conv.eval()
    regressor.eval()
    image_name = os.path.basename(args.input_dir)
    image_name = os.path.splitext(image_name)[0]
    if args.bbox_dir is None:
        out_bbox_file = "{}/{}_box.txt".format(args.output_dir, image_name)
        fout = open(out_bbox_file, "w")
        im = cv2.imread(args.input_dir) # [183, 275, 3]
        cv2.imshow('image', im)
        rects = select_exemplar_rois(im)
        rects1 = list()
        for rect in rects:
            y1, x1, y2, x2 = rect
            rects1.append([y1, x1, y2, x2])
            fout.write("{} {} {} {}\n".format(y1, x1, y2, x2))
        fout.close()
        cv2.destroyWindow("Image")
    else:
        with open(args.bbox_dir, "r") as fin:
            lines = fin.readlines()
        rects1 = list()
        for line in lines:
            data = line.split()
            y1 = int(data[0])
            x1 = int(data[1])
            y2 = int(data[2])
            x2 = int(data[3])
            rects1.append([y1, x1, y2, x2])
    image = Image.open(args.input_dir)
    image.load()
    sample = {'image': image, 'lines_boxes': rects1}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']
    image = image.cuda() # [3, 183, 275]
    boxes = boxes.cuda() # [1, 3, 5]
    with torch.no_grad():
        features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales) # [1, 3, 6, 23, 25]
    if not args.adapt:
        with torch.no_grad():
            output = regressor(features) # [1, 1, 184, 280]
    else:
        features.required_grad = True
        adapted_regressor = regressor
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.lr)
        pbar = tqdm(range(args.gradient_steps))
        for step in pbar:
            optimizer.zero_grad()
            output = adapted_regressor(features) # [1, 1, 184, 280]
            lCount = args.weight_mincount * MincountLoss(output, boxes)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
            Loss = lCount + lPerturbation
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()
            pbar.set_description('Adaptation step: {:<3}, Loss: {:.2f}, Pred: {:.2f}'.format(step, Loss.item(), output.sum().item()))
        features.required_grad = False
        output = adapted_regressor(features) # [1, 1, 184, 280]
    print('Pred:', output.sum().item())
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    rslt_file = "{}/{}_out.png".format(args.output_dir, image_name)
    visualize_output_and_save(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--input_dir", type=str, default='assets/orange.jpg')
    parser.add_argument("--bbox_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='saved_results')
    parser.add_argument("--ckpt_dir", type=str, default="data/pretrainedModels/FamNet_Save1.pth")
    # testing config
    parser.add_argument("--adapt", action='store_true') # test time adaptation
    parser.add_argument("--gradient_steps", type=int, default=100)
    parser.add_argument("-lr", type=float, default=1e-7)
    parser.add_argument("--weight_mincount", type=float, default=1e-9)
    parser.add_argument("--weight_perturbation", type=float, default=1e-4)
    args = parser.parse_args()

    print('Testing image:', args.input_dir)
    main(args)
from __future__ import division
import os
import argparse
import random
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from .models import load_model
from .utils.utils import  provide_determinism
from .utils.utils import load_classes, non_max_suppression, rescale_boxes
from .utils.datasets import RealPredict
from .utils.augmentations import AUGMENTATION_TRANSFORMS
from .utils.transforms import DEFAULT_TRANSFORMS, Resize
from .utils.parse_config import parse_data_config
from .utils.loss import compute_loss
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageFile
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
ImageFile.LOAD_TRUNCATED_IMAGES = True
matplotlib.use('Agg')

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class DetectorTrainer():
    def __init__(self, external_args, logger):
        parser = argparse.ArgumentParser(description="Trains the YOLO model.")
        parser.add_argument("-m", "--model", type=str, default="models/detector/config/yolov3.cfg", help="Path to model definition file (.cfg)")
        parser.add_argument("-d", "--data", type=str, default="models/detector/config/coco.data", help="Path to data config file (.data)")
        parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs")
        parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
        parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
        parser.add_argument("--pretrained_weights", type=str, default="pretrained_models/darknet53.conv.74", help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
        parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
        parser.add_argument("--evaluation_interval", type=int, default=5, help="Interval of epochs between evaluations on validation set")
        parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
        parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
        parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
        parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
        parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
        args = parser.parse_args()
        
        if args.seed != -1:
            provide_determinism(args.seed)
        self.logger = logger
        data_config = parse_data_config(args.data)
        self.class_names = load_classes(data_config["names"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(args.model, os.path.join(external_args.input_dir, "darknet53.conv.74"))
        self.mini_batch_size = self.model.hyperparams['batch'] // self.model.hyperparams['subdivisions']
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.model.hyperparams['optimizer'] in [None, "adam"]:
            self.optimizer = optim.Adam(params, lr=self.model.hyperparams['learning_rate'], weight_decay=self.model.hyperparams['decay'])
        elif self.model.hyperparams['optimizer'] == "sgd":
            self.optimizer = optim.SGD(params, lr=self.model.hyperparams['learning_rate'], weight_decay=self.model.hyperparams['decay'], momentum=self.model.hyperparams['momentum'])
        else:
            print('This optimizer does not exist')
            raise NotImplementedError
        self.train_transform = AUGMENTATION_TRANSFORMS
        self.val_transform = DEFAULT_TRANSFORMS
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.img_size = self.model.hyperparams['height']
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(self.img_size)])
        self.real_stage = ["train", "test"]
        self.real_dataset = {x: RealPredict(external_args.scene_dataset, x, transform) for x in self.real_stage}
        self.real_dataloader = {x: torch.utils.data.DataLoader(self.real_dataset[x], collate_fn=default_collate, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) for x in self.real_stage}
        self.args = args
        external_args.detect_conf = 0.5
        self.external_args = external_args

    def save_model(self, iter_num):
        model_state_dict = self.model.state_dict()
        model_save_path = os.path.join(self.external_args.output_dir, 'models')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model_state_dict, os.path.join(model_save_path, 'detect_model_{}.pth'.format(iter_num)))

    def batch_data(self, images, boxes, start, end, transform, batch_count, multiscale=False):
        batch_images = images[start:end]
        batch_boxes = boxes[start:end]
        batch = []
        for index, (img, img_boxes) in enumerate(zip(batch_images, batch_boxes)):
            img, img_bb_targets = transform((np.array(img), img_boxes))
            batch.append((img, img_bb_targets))
        batch = [data for data in batch if data is not None]
        imgs, bb_targets = list(zip(*batch))
        if multiscale and batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        return  imgs, bb_targets
    
    def train_epoch(self, data):
        self.model.train()
        start = 0
        end = self.mini_batch_size
        images = deepcopy(data['images'])
        boxes = deepcopy(data['gt_boxes'])
        shuffle_index = np.random.permutation(np.arange(len(images)))
        images = [images[ind] for ind in shuffle_index]
        boxes = [boxes[ind] for ind in shuffle_index]
        total_loss_list = []
        while end <= len(images) and start != end:
            self.train_batch_count += 1
            imgs, targets = self.batch_data(images, boxes, start, end, self.train_transform, self.train_batch_count, self.args.multiscale_training)
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)
            outputs = self.model(imgs)
            loss, loss_components = compute_loss(outputs, targets, self.model)
            total_loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            start = end
            end = min(end + self.mini_batch_size, len(images))
        return total_loss_list

    def train(self, data):
        for epoch in range(1, self.args.epochs + 1):
            self.epoch = epoch
            total_loss_list = self.train_epoch(data['train_data'])
            print('Epoch: [{}/{}], Loss: {:.4f}'.format(self.epoch, self.args.epochs + 1, np.mean(total_loss_list)))
    
    def predict(self, iter_num):
        conf_thres = self.external_args.detect_conf
        nms_thres = 0.4
        self.model.eval()
        data = {}
        scale_data = {}
        for stage in self.real_stage:
            key = stage + "_data"
            data[key] = {}
            data[key]['images'] = []
            data[key]['images_names'] = []
            data[key]['gt_boxes'] = []
            data[key]['visual_boxes'] = []
            scale_data[key] = {}
            scale_data[key]['y_positions'] = []
            scale_data[key]['scales'] = []
            for img, img_tensor, name in self.real_dataloader[stage]:
                input_imgs = img_tensor.to(self.device)
                with torch.no_grad():
                    detections = self.model(input_imgs)
                    detections = non_max_suppression(detections, conf_thres, nms_thres)
                data[key]['visual_boxes'].extend(detections)
                labels = detections[0][:,-1]
                boxes = rescale_boxes(deepcopy(detections[0]), self.img_size, img.shape[1:3])
                scale_data[key]['y_positions'].append((boxes[:,1] + (boxes[:,3]-boxes[:,1])*0.1)[:,None].cpu().numpy())
                scale_data[key]['scales'].append(((boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1]))[:,None].cpu().numpy())
                boxes = torch.cat((labels[:,None], (((boxes[:,0] + boxes[:,2]) / 2) / img.shape[2])[:,None], (((boxes[:,1] + boxes[:,3]) / 2) / img.shape[1])[:,None],
                                   ((boxes[:,2] - boxes[:,0]) / img.shape[2])[:,None], ((boxes[:,3] - boxes[:,1]) / img.shape[1])[:,None]), 1)
                img = Image.fromarray(img.squeeze(0).numpy()).convert("RGB")
                data[key]['images'].append(img)
                data[key]['images_names'].append(name[0])
                data[key]['gt_boxes'].append(boxes.cpu().numpy())
        self.draw_and_save_output_images(data, iter_num)
        return data, scale_data
    
    def draw_and_save_output_images(self, data, iter_num):
        images = deepcopy(data['train_data']['images'])
        images_names = deepcopy(data['train_data']['images_names'])
        gt_boxes = deepcopy(data['train_data']['visual_boxes'])
        save_path = os.path.join(self.external_args.output_dir, "detected_show", "iter_{}".format(iter_num))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for index, (img, img_names, img_gt_boxes) in enumerate(zip(images, images_names, gt_boxes)):
            img = np.array(img)
            self.draw_and_save_output_image(img, img_gt_boxes, img_names, save_path)
    
    def draw_and_save_output_image(self, img, img_gt_boxes, img_names, save_path):
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        detections = rescale_boxes(img_gt_boxes, self.img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_pred in detections:
            print(f"\t+ Label: {self.class_names[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
            box_w = x2 - x1
            box_h = y2 - y1
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(bbox)
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        output_path = os.path.join(save_path, f"{img_names}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()
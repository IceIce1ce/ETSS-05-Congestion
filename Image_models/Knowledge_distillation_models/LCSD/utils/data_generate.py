import os
import glob
import cv2
import json
import math
import torch
import pickle
import random
import numpy as np
from PIL import Image
from copy import deepcopy
from torchvision import transforms
from sortedcontainers import SortedDict
from models.har_model.util import util as har_util
from models.har_model.models.networks import RainNet
from models.har_model.models.normalize import RAIN
from utils.utils import generate_gaussian_kernels, compute_distances, get_gt_dots, gaussian_filter_density
import matplotlib
matplotlib.use('Agg') 

class Har:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.har_model = self.load_har_model(args)
        self.har_model = self.har_model.to(self.device)
        self.har_transform_image = transforms.Compose([transforms.Resize([args.har_image_size, args.har_image_size]), transforms.ToTensor(),
                                                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.har_transform_mask = transforms.Compose([transforms.Resize([args.har_image_size, args.har_image_size]), transforms.ToTensor()])
        self.args = args

    def load_har_model(self, args):
        net = RainNet(input_nc=3, output_nc=3, ngf=64, norm_layer=RAIN, use_dropout=True)
        load_path = os.path.join(args.input_dir, "net_G_last.pth")
        assert os.path.exists(load_path), print('%s not exists. Please check the file' % load_path)
        state_dict = torch.load(load_path, map_location='cpu')
        har_util.copy_state_dict(net.state_dict(), state_dict)
        print('Load harmonization model from:', load_path)
        return net

    def harmonization(self, images, masks): # [480, 640, 3], [480, 640]
        batch_size = self.args.har_batch_size
        har_images = []
        img_height = images[0].height
        img_width = images[0].width
        start = 0
        end = min(batch_size, len(images))
        while end <= len(images) and start != end:
            batch_images = images[start:end]
            batch_masks = masks[start:end]
            batch_images_tensor = [self.har_transform_image(img).unsqueeze(0).to(self.device) for img in batch_images]
            batch_masks_tensor = [self.har_transform_mask(mask).unsqueeze(0).to(self.device) for mask in batch_masks]
            batch_images_tensor = torch.cat(batch_images_tensor)
            batch_masks_tensor = torch.cat(batch_masks_tensor)
            batch_har_images_tensor = self.har_model.processImage(batch_images_tensor, batch_masks_tensor, batch_images_tensor)
            batch_har_images = [har_util.tensor2im(batch_har_images_tensor[i].unsqueeze(0))  for i in range(batch_har_images_tensor.shape[0])]
            har_images.extend(batch_har_images)
            del batch_images_tensor, batch_masks_tensor, batch_har_images_tensor, batch_har_images
            torch.cuda.empty_cache()
            start = end
            end = min(end + batch_size, len(images))
        images = [Image.fromarray(img).resize((img_width, img_height), Image.ANTIALIAS) for img in har_images] # [480, 640, 3]
        return images

class DatasetGenerator:
    def __init__(self, args):
        args.min_ped_num = 0
        args.max_ped_num = 100
        args.num_pattern = 'uniform'
        args.pos_pattern = 'uniform'
        args.harmonization = True
        args.har_batch_size = 16
        args.har_image_size = 512
        args.save_synthetic_dataset = True
        args.ped_source = "GCC"
        args.ped_num = 20
        self.stage = ['train']
        if args.harmonization:
            self.har = Har(args)
        pedestrian_dir = os.path.join(args.input_dir, "pedestrians", args.ped_source)
        with open(os.path.join(pedestrian_dir, "info_json.json"), "r") as f:
            self.pesestrians_info = json.load(f)
        self.pedestrians_all = {}
        pedestrians_path = glob.glob(os.path.join(pedestrian_dir, '*.png'))
        for p in pedestrians_path:
            base_name = os.path.basename(p)
            id = base_name.split(".")[0]
            self.pedestrians_all[id] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        select_pedestrians_keys = random.sample(list(self.pedestrians_all), args.ped_num)
        self.pedestrians = {key: self.pedestrians_all[key] for key in select_pedestrians_keys}
        base_path = os.path.join(args.input_dir, args.type_dataset, args.scene, "scene.jpg")
        self.base_image = cv2.cvtColor(cv2.imread(base_path), cv2.COLOR_BGR2RGB)
        negetive_samples_paths = glob.glob(os.path.join(args.input_dir, "*_negetive_samples", "*.jpg"))
        self.negetive_samples = []
        for negetive_samples_path in negetive_samples_paths:
            img = Image.open(negetive_samples_path).convert("RGB")
            self.negetive_samples.append(img)
        self.args = args
        self.bg_area = self.base_image.shape[0]*self.base_image.shape[1]
        self.pre_def_scale = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    
    def generate(self, iter_num, predicted_distribution, predicted_num, scale_model=None):
        self.dataset_name = '_'.join([self.args.type_dataset, self.args.scene, self.args.ped_source + "-{}".format(self.args.ped_num), str(self.args.train_num)])
        if predicted_num is not None:
            self.dataset_name += '_prenum'
        else:
            self.dataset_name += '_{}-{}-{}'.format(str(self.args.min_ped_num), str(self.args.max_ped_num), 'uniform')
        probability_cumulative_array = None
        self.roi_mask = np.ones(self.base_image.shape[0:2], dtype=np.uint8) # [480, 640]
        if isinstance(predicted_distribution, np.ndarray):
            predicted_distribution = cv2.resize(predicted_distribution, (self.base_image.shape[1], self.base_image.shape[0]), cv2.INTER_CUBIC)
            map_sum = np.sum(predicted_distribution)
            probability_map = predicted_distribution / map_sum
            probability_array = probability_map.reshape(probability_map.shape[0] * probability_map.shape[1])
            probability_cumulative_array = np.cumsum(probability_array)
            self.args.predicted_distribution =True
            self.dataset_name += '_predis'
        else:
            self.dataset_name += "_global-uniform"
        if scale_model != None and scale_model.is_use():
            self.dataset_name += "_fit-scale"
        else:
            self.dataset_name += "_random-scale"
        self.dataset_name += "_" + str(iter_num)
        self.save_base_path = os.path.join(self.args.output_dir, "synthetic_datasets", self.dataset_name)
        if not os.path.exists(self.save_base_path):
            os.makedirs(self.save_base_path)
        data = {}
        for method in self.stage:
            if method == 'train':
                pictures_num = self.args.train_num
                tmp_data = method + '_data'
            else:
                pictures_num = self.args.val_num
                tmp_data = method + '_data'
            images = []
            gt_points = []
            gt_boxes = []
            masks = []
            images_head_diameters = []
            for i in range(pictures_num):
                if predicted_num is not None:
                    totoal_pedestrians_num = np.random.randint(self.args.min_ped_num, int(max(predicted_num)) + 1)
                else:
                    totoal_pedestrians_num = np.random.randint(self.args.min_ped_num, self.args.max_ped_num + 1)
                # [480, 640, 3], [42, 2], [42, 5], [480, 640], 42
                image, point, gt_img_boxes, mask, head_diameters = self.generate_one_image(self.base_image.copy(), totoal_pedestrians_num, probability_cumulative_array, scale_model)
                image = Image.fromarray(image).convert("RGB")
                images.append(image)
                gt_points.append(point)
                gt_boxes.append(gt_img_boxes)
                masks.append(Image.fromarray(mask).convert("1"))
                images_head_diameters.append(head_diameters)
            if self.args.harmonization:
                images = self.har.harmonization(images, masks) # [480, 640, 3]
            gt_density_maps = self.generate_density_map(gt_points, images, 5, images_head_diameters) # [480, 640]
            if tmp_data == 'train_data':
                for img in self.negetive_samples:
                    images.append(img)
                    gt_points.append(np.empty((0, 2)))
                    gt_boxes.append(np.empty((0, 5)))
                    width, height = img.size
                    density_map = np.zeros((height, width), dtype=np.float32)
                    gt_density_maps.append(density_map)
            data[tmp_data] = {}
            data[tmp_data]["images"] = images
            data[tmp_data]["gt_points"] = gt_points
            data[tmp_data]["gt_density_maps"] = gt_density_maps
            data[tmp_data]["gt_boxes"] = gt_boxes
            data[tmp_data]['masks'] = []
            for img in images:
                width, height = img.size
                data[tmp_data]['masks'].append(np.ones((height, width), dtype=np.float32))
        if self.args.save_synthetic_dataset: # True
            self.save_synthetic_dataset(data)
        return data, deepcopy(self.base_image) # [480, 640, 3]
    
    def generate_one_image(self, base_image, totoal_pedestrians_num, probability_cumulative_array, scale_model):
        position_cache = np.zeros_like(self.base_image[:, :, 0], dtype=np.int64) # [480, 640]
        dot_cache = np.zeros_like(self.base_image[:, :, 0], dtype=np.int64) # [480, 640]
        count = 0
        ground_truth = []
        gt_img_boxes = []
        head_diameters = []
        pedestrians_keys = list(self.pedestrians.keys())
        for i in range(totoal_pedestrians_num):
            # 107, 0, 6, 0.12, []
            head_point, box, index, scale_rate, head_diameter = self.select_position_pedestrian(pedestrians_keys, dot_cache, probability_cumulative_array, scale_model)
            pedestrain_info = self.pesestrians_info[str(index)]
            head_x = pedestrain_info["ann"]["x"]
            head_y = pedestrain_info["ann"]["y"]
            pedestrian = self.pedestrians[str(index)]
            base_image, position_cache, dot_cache = self.paste(base_image, pedestrian, head_point, head_x, head_y, position_cache, scale_rate, dot_cache) # [480, 640, 3], [480, 640], [480, 640]
            count += 1
            ground_truth.append(head_point)
            gt_img_boxes.append(box)
            head_diameters.append(head_diameter)
        ground_truth = np.array(ground_truth) # [42, 2]
        gt_img_boxes = np.array(gt_img_boxes) # [42, 5]
        if ground_truth.shape[0] == 0:
            ground_truth = np.empty((0, 2))
            gt_img_boxes = np.empty((0, 5))
        else:
            gt_img_boxes[:,[1, 3]] = gt_img_boxes[:, [1, 3]] / base_image.shape[1]
            gt_img_boxes[:,[2, 4]] = gt_img_boxes[:, [2, 4]] / base_image.shape[0]
        position_cache = position_cache.astype(bool)
        position_cache = position_cache.astype(np.uint8) * 255
        return base_image, ground_truth, gt_img_boxes, position_cache, head_diameters
    
    def select_position_pedestrian(self, pedestrians_keys, dot_cache, probability_cumulative_array, scale_model):
        base_height = self.base_image.shape[0]
        base_width = self.base_image.shape[1]
        while 1 :
            if isinstance(probability_cumulative_array, np.ndarray):
                u = np.random.rand()
                while u > probability_cumulative_array[-1]:
                    u = np.random.rand()
                pos_index = np.where(u <= probability_cumulative_array)[0][0]
                y = pos_index // base_width
                x = pos_index - y*base_width
            else:
                x = np.random.randint(1, base_width)
                y = np.random.randint(1, base_height)
            index = np.random.choice(pedestrians_keys, 1)[0]
            pedestrain_info = self.pesestrians_info[str(index)]
            pedestrian_height = pedestrain_info["height"]
            pedestrian_width = pedestrain_info["width"]
            head_x = pedestrain_info["ann"]["x"]
            head_y = pedestrain_info["ann"]["y"]
            if scale_model != None and scale_model.is_use():
                current_area = scale_model.predict(y)
            else:
                random_scale = random.choice(self.pre_def_scale)
                max_pedestrian_area = int(self.bg_area / random_scale)
                current_area = (y / base_height) * max_pedestrian_area * (2 / 3)
            if current_area < 0:
                continue
            rate = math.sqrt(current_area/(pedestrian_height*pedestrian_width))
            pedestrian_height = int(pedestrian_height*rate)
            pedestrian_width = int(pedestrian_width*rate)
            head_x = int(head_x*rate)
            head_y = int(head_y*rate)
            head_diameter = pedestrian_width / 2
            x1  = x - head_x
            y1 = y - head_y
            x2 = x1 + pedestrian_width
            y2 = y1 + pedestrian_height
            if 0 < x1 < base_width and 0 < x2 < base_width and 0 < y1 < base_height and 0 < y2 < base_height \
                and pedestrian_height != 0 and pedestrian_width != 0 and self.roi_mask[y, x] == 1 \
                and dot_cache[y, x] == 0 and current_area >= 10:
                break
        assert rate != 0
        return [x, y], [0, x1 + pedestrian_width / 2, y1 + pedestrian_height / 2, pedestrian_width, pedestrian_height], index, rate, head_diameter
    
    def paste(self, base_image, pedestrian, head_point, head_x, head_y, position_cache, scale_rate, dot_cache):
        p_height = int(pedestrian.shape[0] * scale_rate)
        p_width = int(pedestrian.shape[1] * scale_rate)
        pedestrian =  cv2.resize(pedestrian, (p_width, p_height)) # [19, 9, 3]
        head_x = int(head_x*scale_rate)
        head_y = int(head_y*scale_rate)
        x = head_point[0]
        y = head_point[1]
        x_left = x - head_x
        y_top = y - head_y
        mask = pedestrian.copy()
        mask = mask.astype(bool)
        figure_mask = mask.astype(np.int64)[:, :, 0]
        mask = np.invert(mask)
        bk_mask = mask.astype(np.int64)[:,:,0]
        position_mask = np.ones_like(mask[:, :, 0]) * y
        local_position_cache = position_cache[y_top:y_top + p_height, x_left:x_left + p_width]
        indicate_mask = position_mask - local_position_cache
        indicate_mask[indicate_mask >= 0] = 0
        indicate_mask[indicate_mask < 0] = 1
        indicate_mask = indicate_mask*figure_mask
        indicate_mask = indicate_mask + bk_mask
        indicate_mask = indicate_mask.astype(np.int64)
        position_cache[y_top:y_top+p_height,x_left:x_left + p_width] *=  indicate_mask
        invert_indicate_mask = np.invert(indicate_mask.astype(bool))
        invert_indicate_mask = invert_indicate_mask.astype(np.int64)
        position_cache[y_top:y_top+p_height,x_left:x_left+p_width] += invert_indicate_mask*y
        dot_cache[y_top:y_top+int(p_width / 2), x_left:x_left+p_width] += figure_mask[0:int(p_width / 2), :]
        indicate_mask = indicate_mask.reshape(indicate_mask.shape[0], indicate_mask.shape[1],1)
        three_channel_indicate_mask = np.concatenate((indicate_mask, indicate_mask, indicate_mask), axis=2)
        invert_indicate_mask = invert_indicate_mask.reshape(invert_indicate_mask.shape[0], invert_indicate_mask.shape[1], 1)
        three_channel_invert_indicate_mask = np.concatenate((invert_indicate_mask, invert_indicate_mask, invert_indicate_mask), axis=2)
        base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] = base_image[y_top:y_top+p_height,x_left:x_left+p_width,:]*three_channel_indicate_mask
        base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] = base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] + pedestrian*three_channel_invert_indicate_mask
        return base_image, position_cache, dot_cache # [480, 640, 3], [480, 640], [480, 640]

    def generate_density_map(self, points, images, sigma_method, images_head_diameters):
        gt_density_maps = []
        precomputed_kernels_path = os.path.join(self.args.output_dir, 'gaussian_kernels.pkl')
        if not os.path.exists(precomputed_kernels_path):
            generate_gaussian_kernels(precomputed_kernels_path, round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=128, num_sigmas=129)
        with open(precomputed_kernels_path, 'rb') as f:
            kernels_dict = pickle.load(f)
            kernels_dict = SortedDict(kernels_dict)
        precomputed_distances_path = os.path.join(self.save_base_path, 'distances_dict.pkl')
        compute_distances(precomputed_distances_path, images, points)
        with open(precomputed_distances_path, 'rb') as f:
            distances_dict = pickle.load(f)
        for index, (img, point, img_head_diameters) in enumerate(zip(images, points, images_head_diameters)):
            width, height = img.size
            gt_points = get_gt_dots(point, height, width) # [42, 2]
            distances = distances_dict[index] # [42, 4]
            density_map = gaussian_filter_density(gt_points, height, width, distances, kernels_dict, img_head_diameters, min_sigma=2, method=sigma_method, const_sigma=15)
            gt_density_maps.append(density_map)
        return gt_density_maps

    def save_synthetic_dataset(self, data):
        for key, value  in data.items():
            images = value['images']
            gt_points = value['gt_points']
            gt_density_maps = value['gt_density_maps']
            save_images_path = os.path.join(self.save_base_path, key, 'images')
            if not os.path.exists(save_images_path):
                os.makedirs(save_images_path)
            save_gt_points_path = save_images_path.replace("images", "gt_points")
            if not os.path.exists(save_gt_points_path):
                os.makedirs(save_gt_points_path)
            save_gt_density_map_path = save_images_path.replace("images", "gt_density_maps")
            if not os.path.exists(save_gt_density_map_path):
                os.makedirs(save_gt_density_map_path)
            image_txt_pairs = []
            for i, img in enumerate(images):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # [480, 640, 3]
                gt_point = np.array(gt_points[i]) # [42, 2]
                gt_density_map = gt_density_maps[i] # [480, 640]
                img_save_path = os.path.join(save_images_path, str(i+1)+".jpg")
                cv2.imwrite(img_save_path, img)
                gt_save_path = img_save_path.replace("images", "gt_points").replace("jpg", "txt")
                np.savetxt(gt_save_path, gt_point, fmt="%d")
                gt_density_map_save_path = img_save_path.replace("images", "gt_density_maps").replace("jpg", "npy")
                np.save(gt_density_map_save_path, gt_density_map)
                image_txt_pairs.append([img_save_path.replace(self.save_base_path + '/', ""), gt_save_path.replace(self.save_base_path + '/', ""),
                                        gt_density_map_save_path.replace(self.save_base_path + '/', "")])
            image_txt_pairs = np.array(image_txt_pairs)
            np.savetxt(os.path.join(self.save_base_path, key + '.list'), image_txt_pairs, fmt="%s")
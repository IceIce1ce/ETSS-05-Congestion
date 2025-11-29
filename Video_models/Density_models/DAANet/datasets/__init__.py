import os
import misc.transforms as own_transforms
from  misc.transforms import  check_image
import torchvision.transforms as standard_transforms
from . import dataset
from . import setting
from . import samplers
from torch.utils.data import DataLoader
import random

class train_pair_transform(object):
    def __init__(self,cfg_data, check_dim = True):
        self.cfg_data = cfg_data
        self.pair_flag = 0
        self.scale_factor = 1
        self.last_cw_ch =(0,0)
        self.crop_left = (0,0)
        self.last_crop_left = (0, 0)
        self.rate_range = cfg_data.crop_rate
        self.resize_and_crop = own_transforms.RandomCrop()
        self.scale_to_setting = own_transforms.ScaleByRateWithMin(cfg_data.train_size[1], cfg_data.train_size[0])
        self.flip_flag = 0
        self.horizontal_flip = own_transforms.RandomHorizontallyFlip()
        self.last_frame_size = (0,0)
        self.check_dim = check_dim

    def __call__(self,img,target):
        w_ori, h_ori = img.size
        if self.pair_flag == 1 and self.check_dim:
            assert self.last_frame_size == (w_ori,w_ori)
        if self.pair_flag % 2 == 0:
            self.scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])
            self.c_h,self.c_w = int(self.cfg_data.train_size[0] / self.scale_factor), int(self.cfg_data.train_size[1] / self.scale_factor)
            img, target = check_image(img, target, (self.c_h,self.c_w))
            w, h = img.size
            self.last_cw_ch = (self.c_w, self.c_h)
            self.pair_flag = 0
            self.last_frame_size = (w_ori, w_ori)
            x1 = random.randint(0, w - self.c_w)
            y1 = random.randint(0, h - self.c_h)
            self.last_crop_left = (x1,y1)
            self.flip_flag = round(random.random())
        else:
            img, target = check_image(img, target, (self.c_h, self.c_w))
        img, target = self.resize_and_crop(img, target, self.crop_left,crop_size=(self.c_h, self.c_w))
        img, target = self.scale_to_setting(img, target)
        img, target = self.horizontal_flip(img, target, self.flip_flag)
        self.pair_flag += 1
        return img, target # [768, 1024, 3], [1]

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

def createTrainData(datasetname, Dataset, cfg, cfg_data):
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*cfg.mean_std)])
    main_transform = train_pair_transform(cfg)
    train_set = Dataset(cfg_data.TRAIN_LST, cfg_data.DATA_PATH, main_transform=main_transform, img_transform=img_transform, train=True, datasetname=datasetname)
    train_sampler = samplers.CategoriesSampler(train_set.labels, frame_intervals=cfg.train_frame_intervals, n_per=cfg.train_batch_size)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    print('Number of training image:', train_set.__len__())
    return train_loader

def createRestore(mean_std):
    return standard_transforms.Compose([own_transforms.DeNormalize(*mean_std), standard_transforms.ToPILImage()])

def loading_data(cfg):
    datasetname = cfg.type_dataset.upper()
    cfg_data = getattr(setting, datasetname).cfg_data
    train_Dataset = dataset.Dataset
    train_loader = createTrainData(datasetname, train_Dataset, cfg, cfg_data)
    restore_transform = createRestore(cfg.mean_std)
    Dataset = dataset.TestDataset
    val_loader = createValTestData(datasetname, Dataset, cfg, cfg_data, mode='val')
    return train_loader, val_loader, restore_transform

def createValTestData(datasetname, Dataset, cfg, cfg_data, mode ='val'):
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*cfg.mean_std)])
    target = True
    if mode == 'test':
        lst = cfg_data.TEST_LST
        if datasetname=='HT21':
            target = False
    else:
        lst = cfg_data.VAL_LST
    with open(os.path.join( cfg_data.DATA_PATH, lst), 'r') as txt:
        scene_names = txt.readlines()
        scene_names = [i.strip() for i in scene_names]
    data_loader = []
    for scene_name in scene_names:
        sub_dataset = Dataset(scene_name=scene_name, base_path=cfg_data.DATA_PATH, img_transform=img_transform, interval=cfg.val_intervals, target=target, datasetname=datasetname)
        sub_loader = DataLoader(sub_dataset, batch_size=cfg.val_batch_size, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        data_loader.append(sub_loader)
    return data_loader

def loading_testset(cfg, mode='test'):
    datasetname = cfg.type_dataset.upper()
    cfg_data = getattr(setting, datasetname).cfg_data
    Dataset = dataset.TestDataset
    test_loader = createValTestData(datasetname, Dataset, cfg, cfg_data, mode=mode)
    restore_transform = createRestore(cfg.mean_std)
    return test_loader, restore_transform
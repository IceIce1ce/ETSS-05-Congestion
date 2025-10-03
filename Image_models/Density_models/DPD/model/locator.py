from model.HR_Net.seg_hrnet import get_seg_model
from model.VGG.VGG16_FPN import VGG16_FPN
import torch.nn as nn
from model.PBM import BinarizedModule
import torch
import torch.nn.functional as F

class Crowd_locator(nn.Module):
    def __init__(self, net_name, gpu_id):
        super(Crowd_locator, self).__init__()
        if net_name == 'HR_Net':
            self.Extractor = get_seg_model()
            self.Binar = BinarizedModule(input_channels=720)
        elif net_name == 'VGG16_FPN':
            self.Extractor = VGG16_FPN()
            self.Binar = BinarizedModule(input_channels=768)
        else:
            print('This model does not exist')
            raise NotImplementedError
        if len(gpu_id) > 1:
            self.Extractor = torch.nn.DataParallel(self.Extractor).cuda()
            self.Binar = torch.nn.DataParallel(self.Binar).cuda()
        else:
            self.Extractor = self.Extractor.cuda()
            self.Binar = self.Binar.cuda()
        self.loss_BCE = nn.BCELoss().cuda()

    @property
    def loss(self):
        return  self.head_map_loss, self.binar_map_loss

    def forward(self, img, mask_gt, mode='train'):
        feature, pre_map = self.Extractor(img)
        threshold_matrix, binar_map = self.Binar(feature, pre_map)
        if mode == 'train':
            assert pre_map.size(2) == mask_gt.size(2)
            self.binar_map_loss = (torch.abs(binar_map-mask_gt)).mean()
            self.head_map_loss = F.mse_loss(pre_map, mask_gt)
        if mode == 'pseudo':
            return threshold_matrix, pre_map, binar_map, feature
        return threshold_matrix, pre_map, binar_map
import torch

class get_ROI_and_MatchInfo(object):
    def __init__(self,train_size,rdius=8,feature_scale=0.125):
        self.h = train_size[0]
        self.w = train_size[1]
        self.radius = rdius
        self.feature_scale = feature_scale

    def __call__(self, target_a, target_b,noise=None, shape =None):
        gt_a, gt_b = target_a['points'], target_b['points'] # [111, 2], [110, 2]
        if shape is not None:
            self.h = shape[0]
            self.w = shape[1]
        if noise == 'ab':
            gt_a, gt_b = gt_a + torch.randn(gt_a.size()).to(gt_a)*2, gt_b + torch.randn(gt_b.size()).to(gt_b)*2
        elif noise == 'a':
            gt_a = gt_a + torch.randn(gt_a.size()).to(gt_a)
        elif noise == 'b':
            gt_b = gt_b + torch.randn(gt_b.size()).to(gt_b)
        roi_a = torch.zeros(gt_a.size(0), 5).to(gt_a)
        roi_b = torch.zeros(gt_b.size(0), 5).to(gt_b)
        roi_a[:, 0] = 0
        roi_a[:, 1] = torch.clamp(gt_a[:, 0] - self.radius,min=0)
        roi_a[:, 2] = torch.clamp(gt_a[:, 1] - self.radius, min=0)
        roi_a[:, 3] = torch.clamp(gt_a[:, 0] + self.radius, max=self.w)
        roi_a[:, 4] = torch.clamp(gt_a[:, 1] + self.radius, max=self.h)
        roi_b[:, 0] = 1
        roi_b[:, 1] = torch.clamp(gt_b[:, 0] - self.radius, min=0)
        roi_b[:, 2] = torch.clamp(gt_b[:, 1] - self.radius, min=0)
        roi_b[:, 3] = torch.clamp(gt_b[:, 0] + self.radius, max=self.w)
        roi_b[:, 4] = torch.clamp(gt_b[:, 1] + self.radius, max=self.h)
        pois = torch.cat([roi_a, roi_b], dim=0)
        a_ids = target_a['person_id']
        b_ids = target_b['person_id']
        dis = a_ids.unsqueeze(1).expand(-1,len(b_ids)) - b_ids.unsqueeze(0).expand(len(a_ids),-1)
        dis = dis.abs()
        matched_a, matched_b = torch.where(dis==0)
        matched_a2b = torch.stack([matched_a, matched_b],1) # [100, 2]
        unmatched0 = torch.where(dis.min(1)[0] > 0)[0] # [11]
        unmatched1 = torch.where(dis.min(0)[0] > 0)[0] # [10]
        match_gt = {'a2b': matched_a2b, 'un_a': unmatched0, 'un_b': unmatched1}
        return  match_gt, pois
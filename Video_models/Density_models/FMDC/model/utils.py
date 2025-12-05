# https://huggingface.co/stevetod/doduo/blob/main/utils.py
import torch
import torch.nn.functional as F
from .position import PositionEmbeddingSine

def normalize_img(img0, img1):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std
    return img0, img1

def split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)
    else:
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)
    return feature

def merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)
    else:
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)
    return merge

def generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w, shift_size_h, shift_size_w, device=torch.device('cuda')):
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)
    h_slices = (slice(0, -window_size_h), slice(-window_size_h, -shift_size_h), slice(-shift_size_h, None))
    w_slices = (slice(0, -window_size_w), slice(-window_size_w, -shift_size_w), slice(-shift_size_w, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)
    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    if attn_splits > 1:
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)
        position = pos_enc(feature0_splits)
        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position
        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)
        feature0 = feature0 + position
        feature1 = feature1 + position
    return feature0, feature1

def upsample_flow_with_mask(flow, up_mask, upsample_factor, is_depth=False):
    mask = up_mask
    b, flow_channel, h, w = flow.shape
    mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)
    mask = torch.softmax(mask, dim=2)
    multiplier = 1 if is_depth else upsample_factor
    up_flow = F.unfold(multiplier * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    up_flow = up_flow.reshape(b, flow_channel, upsample_factor * h, upsample_factor * w)
    return up_flow

def split_feature_1d(feature, num_splits=2):
    b, w, c = feature.size()
    assert w % num_splits == 0
    b_new = b * num_splits
    w_new = w // num_splits
    feature = feature.view(b, num_splits, w // num_splits, c).view(b_new, w_new, c)
    return feature

def merge_splits_1d(splits, h, num_splits=2):
    b, w, c = splits.size()
    new_b = b // num_splits // h
    splits = splits.view(new_b, h, num_splits, w, c)
    merge = splits.view(new_b, h, num_splits * w, c)
    return merge

def window_partition_1d(x, window_size_w):
    B, W, C = x.shape
    x = x.view(B, W // window_size_w, window_size_w, C).view(-1, window_size_w, C)
    return x

def generate_shift_window_attn_mask_1d(input_w, window_size_w, shift_size_w, device=torch.device('cuda')):
    img_mask = torch.zeros((1, input_w, 1)).to(device)
    w_slices = (slice(0, -window_size_w), slice(-window_size_w, -shift_size_w), slice(-shift_size_w, None))
    cnt = 0
    for w in w_slices:
        img_mask[:, w, :] = cnt
        cnt += 1
    mask_windows = window_partition_1d(img_mask, window_size_w)
    mask_windows = mask_windows.view(-1, window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
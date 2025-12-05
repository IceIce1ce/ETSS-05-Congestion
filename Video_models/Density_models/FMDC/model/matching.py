# https://huggingface.co/stevetod/doduo/blob/refs%2Fpr%2F2/matching.py
import torch
import torch.nn.functional as F
from .geometry import coords_grid, generate_window_grid, normalize_coords

def global_correlation_softmax(feature0, feature1, pred_bidir_flow=False):
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)
    feature1 = feature1.view(b, c, -1)
    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)
    init_grid = coords_grid(b, h, w).to(correlation.device)
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)
    correlation = correlation.view(b, h * w, h * w)
    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)
        init_grid = init_grid.repeat(2, 1, 1, 1)
        grid = grid.repeat(2, 1, 1)
        b = b * 2
    prob = F.softmax(correlation, dim=-1)
    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)
    flow = correspondence - init_grid
    return flow, prob

def local_correlation_softmax(feature0, feature1, local_radius, padding_mode='zeros'):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius, -local_radius, local_radius, local_h, local_w, device=feature0.device)
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid
    sample_coords_softmax = sample_coords
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    sample_coords_norm = normalize_coords(sample_coords, h, w)
    window_feature = F.grid_sample(feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True).permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)
    corr[~valid] = -1e9
    prob = F.softmax(corr, -1)
    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(b, h, w, 2).permute(0, 3, 1, 2)
    flow = correspondence - coords_init
    match_prob = prob
    return flow, match_prob

def local_correlation_with_flow(feature0, feature1, flow, local_radius, padding_mode='zeros', dilation=1):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius, -local_radius, local_radius, local_h, local_w, device=feature0.device)
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid * dilation
    if not isinstance(flow, float):
        sample_coords = sample_coords + flow.view(b, 2, -1).permute(0, 2, 1).unsqueeze(-2)
    else:
        assert flow == 0.
    sample_coords_norm = normalize_coords(sample_coords, h, w)
    window_feature = F.grid_sample(feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True).permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)
    corr = corr.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
    return corr

def global_correlation_softmax_stereo(feature0, feature1):
    b, c, h, w = feature0.shape
    x_grid = torch.linspace(0, w - 1, w, device=feature0.device)
    feature0 = feature0.permute(0, 2, 3, 1)
    feature1 = feature1.permute(0, 2, 1, 3)
    correlation = torch.matmul(feature0, feature1) / (c ** 0.5)
    mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(feature0)
    valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
    correlation[~valid_mask] = -1e9
    prob = F.softmax(correlation, dim=-1)
    correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)
    disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence
    return disparity.unsqueeze(1), prob

def local_correlation_softmax_stereo(feature0, feature1, local_radius):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1).contiguous()
    local_h = 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(0, 0, -local_radius, local_radius, local_h, local_w, device=feature0.device)
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid
    sample_coords_softmax = sample_coords
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    sample_coords_norm = normalize_coords(sample_coords, h, w)
    window_feature = F.grid_sample(feature1, sample_coords_norm, padding_mode='zeros', align_corners=True).permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).contiguous().view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)
    corr[~valid] = -1e9
    prob = F.softmax(corr, -1)
    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()
    flow = correspondence - coords_init
    match_prob = prob
    flow_x = -flow[:, :1]
    return flow_x, match_prob

def correlation_softmax_depth(feature0, feature1, intrinsics, pose, depth_candidates, depth_from_argmax=False, pred_bidir_depth=False):
    b, c, h, w = feature0.size()
    assert depth_candidates.dim() == 4
    scale_factor = c ** 0.5
    if pred_bidir_depth:
        feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)
        intrinsics = intrinsics.repeat(2, 1, 1)
        pose = torch.cat((pose, torch.inverse(pose)), dim=0)
        depth_candidates = depth_candidates.repeat(2, 1, 1, 1)
    warped_feature1 = warp_with_pose_depth_candidates(feature1, intrinsics, pose, 1. / depth_candidates)
    correlation = (feature0.unsqueeze(2) * warped_feature1).sum(1) / scale_factor
    match_prob = F.softmax(correlation, dim=1)
    if depth_from_argmax:
        index = torch.argmax(match_prob, dim=1, keepdim=True)
        depth = torch.gather(depth_candidates, dim=1, index=index)
    else:
        depth = (match_prob * depth_candidates).sum(dim=1, keepdim=True)
    return depth, match_prob

def warp_with_pose_depth_candidates(feature1, intrinsics, pose, depth, clamp_min_depth=1e-3):
    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4
    b, d, h, w = depth.size()
    c = feature1.size(1)
    with torch.no_grad():
        grid = coords_grid(b, h, w, homogeneous=True, device=depth.device)
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(1, 1, d, 1) * depth.view(b, 1, d, h * w)
        points = points + pose[:, :3, -1:].unsqueeze(-1)
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(b, 3, d, h * w)
        pixel_coords = points[:, :2] / points[:, -1:].clamp(min=clamp_min_depth)
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1
        grid = torch.stack([x_grid, y_grid], dim=-1)
    warped_feature = F.grid_sample(feature1, grid.view(b, d * h, w, 2), mode='bilinear', padding_mode='zeros', align_corners=True).view(b, c, d, h, w)
    return warped_feature
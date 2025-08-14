import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter

def gaussian_filter_density(gt, gamma=3, k=4, adaptive=False, mask=None): # [360, 640]
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=k)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if adaptive:
            if gt_count > 1:
                sigma = (np.array([distances[i][j] for j in range(1, k)])) * 0.1
            else:
                sigma = np.average(np.array(gt.shape)) / 2. / 2
        else:
            sigma = gamma
        map = gaussian_filter(pt2d, sigma, mode='constant')
        if mask is not None:
            map *= mask
        map = map / map.sum()
        density += map
    return density # [360, 640]
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.clustering import clustering
from scipy.optimize import linear_sum_assignment

def calc_mean_std(feat, eps=1e-5): # [32, 256, 56, 56]
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps # [32, 256]
    feat_std = feat_var.sqrt().view(N, C) # [32, 256]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C) # [32, 256]
    return feat_mean, feat_std

def reassign(y_before, y_pred): # [3609], [3609]
    assert y_before.size == y_pred.size
    D = max(y_before.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64) # [4, 4]
    for i in range(y_before.size):
        w[y_before[i], y_pred[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return col_ind

def compute_features(dataloader, model, N):
    model.eval()
    for i, (input_tensor, _, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.cuda()
            aux = model.domain_features(input_var).data.cpu().numpy()
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')
            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                features[i * dataloader.batch_size:] = aux.astype('float32')
    return features

def compute_instance_stat(dataloader, model, N):
    model.eval()
    for i, (fname, input_tensor) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.cuda() # [32, 3, 224, 224]
            conv_feats = model.conv_features(input_var) # [1, 32, 256, 56, 56]
            for j, feats in enumerate(conv_feats):
                feat_mean, feat_std = calc_mean_std(feats) # [32, 256], [32, 256]
                if j == 0:
                    aux = torch.cat((feat_mean, feat_std), 1).data.cpu().numpy() # [32, 512]
                else:
                    aux = np.concatenate((aux, torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()), axis=1)
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32') # [3609, 512]
            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                features[i * dataloader.batch_size:] = aux.astype('float32')
    return features

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes) # [3609]
    return np.asarray(pseudolabels)[indexes] # [3609]

def domain_split(dataset, model, cluster_before, nmb_cluster=3, method='Kmeans', pca_dim=256, batchsize=32, num_workers=32, whitening=False, L2norm=False, instance_stat=True):
    cluster_method = clustering.__dict__[method](nmb_cluster, pca_dim, whitening, L2norm)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    if instance_stat:
        features = compute_instance_stat(dataloader, model, len(dataset)) # [3609, 512]
    else:
        features = compute_features(dataloader, model, len(dataset))
    clustering_loss = cluster_method.cluster(features, verbose=False) # None
    cluster_list = arrange_clustering(cluster_method.images_lists)
    mapping = reassign(cluster_before, cluster_list)
    cluster_reassign = [cluster_method.images_lists[mapp] for mapp in mapping]
    return arrange_clustering(cluster_reassign)
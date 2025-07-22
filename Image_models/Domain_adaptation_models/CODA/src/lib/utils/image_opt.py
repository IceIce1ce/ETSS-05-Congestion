import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage import transform as sk_transform

def getPerspective(dots, pmap):
    persp = []
    for i in range(len(dots)):
        x,y = dots[i].astype(np.int32)
        persp.append(pmap[y, x] * 0.185)
    return torch.FloatTensor(np.array(persp))

def findSigma(dots,k,beta):
    PN2 = torch.FloatTensor(dots)
    AB = torch.mm(PN2,torch.t(PN2))
    AA = torch.unsqueeze(torch.diag(AB),1)
    DIST = torch.sqrt(AA - 2 * AB + AA.t())
    sorted,indices = torch.sort(DIST)
    sigma = beta * torch.mean(sorted[:,1:1+k],1)
    return sigma

def fspecial(shape=(3, 3), sigma=0.5):
    m, n = (shape[0] - 1.) / 2., (shape[1] -1) / 2.
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    sumh = h.sum()
    if sumh != 0:
       h /= sumh
    return h

def genDensity(image, dots, sigmas, margin_size, rescale=0.125):
    h, w = image.shape[:2]
    dmap_extend = np.zeros((h + margin_size - 1, w + margin_size - 1), np.float32)
    margin = int((margin_size - 1) / 2)
    for i in range(len(sigmas)):
        cx, cy = dots[i].astype(np.int32)
        sigma = sigmas[i]
        kernel_size = int(min(1 * sigma, margin))
        gaussian_kernel = fspecial((kernel_size, kernel_size), sigma).cpu().detach().numpy()
        dmap_extend[cy + margin:cy + margin + kernel_size, cx + margin:cx + margin + kernel_size] += gaussian_kernel
    dmap = dmap_extend[margin:margin + h, margin:margin + w]
    if rescale != 1.0:
        dmap = sk_transform.resize(dmap, (int(h * rescale), int(w * rescale)), preserve_range=True)
        dmap /= (rescale ** 2)
    return dmap

def getAttentionDensity(image,nlevel, dots,  sigmas, margin_size,rescale = 0.125):
    h, w = image.shape[:2]
    dmap = []
    levels = [3, 9, 27]
    for i in range(nlevel):
        level = levels[i]
        dmap_extend = np.zeros((h + margin_size - 1, w + margin_size - 1), np.float32)
        margin = int((margin_size - 1) / 2)
        for j in range(len(dots)):
            cx, cy = dots[j].astype(np.int32)
            att = sigmas[j][i]
            kernel_size = int(min(level, margin))
            gaussian_kernel = fspecial((kernel_size, kernel_size), level)
            dmap_extend[cy + margin:cy + margin + kernel_size, cx + margin:cx + margin + kernel_size] += gaussian_kernel*att
        sub_dmap =  dmap_extend[margin:margin + h, margin:margin + w]
        if rescale != 1.0:
            sub_dmap = sk_transform.resize(dmap, (int(h * rescale), int(w * rescale)), preserve_range=True)
            sub_dmap /= (rescale ** 2)
        dmap.append(sub_dmap)
    return np.array(dmap)

def getLevel(nlevel, theta,thresholds,dots,k):
    PN2 = torch.FloatTensor(dots)
    AB = torch.mm(PN2, torch.t(PN2))
    AA = torch.unsqueeze(torch.diag(AB), 1)
    DIST = torch.sqrt(AA - 2 * AB + AA.t())
    sorted, indices = torch.sort(DIST)
    d = 0.2 * torch.mean(sorted[:, 1:1 + k], 1)
    v_dots = []
    for i in d:
        th = nlevel
        for j in range(nlevel):
            if thresholds[j]>= i:
                th = j + 1
                break
        v_dot = []
        for l in range(1, nlevel+1):
            tmp = (l - th) * (l - th) / (2 * theta)
            v_dot.append(np.exp(-tmp))
        v_dot = np.array(v_dot)
        v_dot = v_dot/np.sum(v_dot)
        v_dots.append(v_dot)
    return torch.FloatTensor(v_dots)

def showTest(image, pre_dens, gt_dens, idx, save_path):
    for i in range(image.shape[0]):
        img = torch.squeeze(image[i, :], 0)
        img = (img * 127.5 + 127.5).numpy().astype(np.uint8)
        img = img.transpose((1, 2, 0))
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_title('Original Image', fontsize=14)
        ax.imshow(img)
        ax = fig.add_subplot(223)
        gt_densmap = torch.squeeze(gt_dens[i, :], 0)
        ax.set_title('GT: {:.2f}'.format(np.sum(gt_densmap.numpy())))
        ax.imshow(gt_densmap.numpy(), cmap='jet', interpolation='bilinear')
        ax = fig.add_subplot(224)
        pre_densmap = torch.squeeze(pre_dens[i,:],0)
        ax.set_title('Pred: {:.2f}'.format(np.sum(pre_densmap.numpy())))
        ax.imshow(pre_densmap.numpy(), cmap='jet', interpolation='bilinear')
        vis_path = os.path.join(save_path, 'vis')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        plt.savefig(f'{vis_path}/{idx}.png')
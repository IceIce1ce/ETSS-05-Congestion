import torch
import numpy as np

def Class2Count(pre_cls,label_indice):
    if isinstance(label_indice,np.ndarray):
        label_indice = torch.from_numpy(label_indice)
    label_indice = label_indice.squeeze()
    IF_ret_gpu = (pre_cls.device.type == 'cuda')
    label2count = [0.0]
    for (i,item) in enumerate(label_indice):
        if i<label_indice.size()[0]-1:
            tmp_count = (label_indice[i]+label_indice[i+1])/2
        else:
            tmp_count = label_indice[i]
        label2count.append(tmp_count)
    label2count = torch.tensor(label2count)
    label2count = label2count.type(torch.FloatTensor)
    ORI_SIZE = pre_cls.size()
    pre_cls = pre_cls.reshape(-1).cpu()
    pre_counts = torch.index_select(label2count,0,pre_cls.cpu().type(torch.LongTensor))
    pre_counts = pre_counts.reshape(ORI_SIZE)
    if IF_ret_gpu:
        pre_counts = pre_counts.cuda()
    return pre_counts
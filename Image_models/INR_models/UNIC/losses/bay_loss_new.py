from torch.nn.modules import Module
import torch

class Bay_Loss(Module):
    def __init__(self, use_background):
        super(Bay_Loss, self).__init__()
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):
            if prob is None:
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32).cuda()
                loss += torch.sum(torch.abs(target - pre_count))
            else:
                N = len(prob)
                n_sample = target_list[0].shape[1]
                if self.use_bg:
                    target = torch.zeros((N, n_sample), dtype=torch.float32).cuda()
                    target[:-1] = target_list[idx][:-2]
                    pre_count = pre_density[idx].reshape((1, -1)).unsqueeze(0) * prob
                else:
                    target = target_list[idx][:-1]
                    pre_count = pre_density[idx].view((1, -1)).unsqueeze(0) * prob
            loss += torch.nn.MSELoss(reduction = 'sum')(target*10, pre_count[0])
        loss = loss / len(prob_list)
        return loss
import torch
import torch.nn as nn

class NoWeightingStrategy(nn.Module):
    def __init__(self, **kwargs):
        super(NoWeightingStrategy, self).__init__()

    def forward(self, task_losses):
        return task_losses

class UncertaintyWeightingStrategy(nn.Module):
    def __init__(self, tasks):
        super(UncertaintyWeightingStrategy, self).__init__()
        self.tasks = tasks
        self.log_vars = nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, task_losses):
        losses_tensor = torch.stack(list(task_losses.values()))
        non_zero_losses_mask = losses_tensor != 0.0
        losses_tensor = torch.exp(-self.log_vars) * losses_tensor + self.log_vars
        losses_tensor *= non_zero_losses_mask
        weighted_task_losses = task_losses.copy()
        weighted_task_losses.update(zip(weighted_task_losses, losses_tensor))
        return weighted_task_losses
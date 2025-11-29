import torch
from collections import deque

class Task_KPI_Pool:
    def __init__(self,task_setting, maximum_sample):
        self.pool_size = maximum_sample
        self.maximum_sample = maximum_sample
        assert self.pool_size > 0
        self.current_sample = {x: 0 for x in task_setting.keys()}
        self.store = task_setting
        for key, data in self.store.items():
            self.store[key] = {x: deque() for x in data}

    def add(self, save_dict):
        for task_key, data in save_dict.items():
            if self.current_sample[task_key] < self.pool_size:
                self.current_sample[task_key] = self.current_sample[task_key] + 1
                for data_key, data_val in data.items():
                    self.store[task_key][data_key].append(data_val)
            else:
                for data_key, data_val in data.items():
                    self.store[task_key][data_key].popleft()
                    self.store[task_key][data_key].append(data_val)

    def query(self):
        task_KPI = {}
        for task_key in self.store:
            data_keys = list(self.store[task_key].keys())
            gt_list = list(self.store[task_key][data_keys[0]])
            correct_list = list(self.store[task_key][data_keys[1]])
            gt_sum = torch.tensor(gt_list).sum()
            correct_sum = torch.tensor(correct_list).sum()
            task_KPI.update({task_key: correct_sum / (gt_sum + 1e-8)})
        return  task_KPI
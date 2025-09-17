import importlib
import torch.utils.data
from dataset.base_dataset import BaseDataset
import numpy as np

def find_dataset_using_name(dataset_name):
    dataset_filename = "dataset." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    return dataset

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def min_collater(data):
    keys_of_items = data[0].keys()
    imgs = [s['img'] for s in data]
    batches = {}
    for ind in keys_of_items:
        batches[ind] = [s[ind] for s in data]
    widths = [int(s.size(2)) for s in imgs]
    heights = [int(s.size(1)) for s in imgs]
    batch_size = len(imgs)
    min_width = np.array(widths).min()
    min_height = np.array(heights).min()
    padded_batches = {}
    for ind in keys_of_items:
        padded_data = torch.zeros(batch_size, batches[ind][0].size(0), min_height, min_width)
        for batch_ind in range(batch_size):
            data_org = batches[ind][batch_ind]
            padded_data[batch_ind, :, :, :] = data_org[:, :min_height, :min_width]
        padded_batches[ind] = padded_data
    return padded_batches

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        if opt.batch_size > 1 and opt.pad_batch:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=min_collater, num_workers=int(opt.num_threads))
        else:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def get_preprocess_func_for_test(self): 
        return self.dataset.get_preprocess_func_for_test()

    def get_test_imgs(self, img_list):
        return self.dataset.get_test_imgs(img_list)

def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset
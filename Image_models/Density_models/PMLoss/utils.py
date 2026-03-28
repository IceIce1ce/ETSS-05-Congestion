import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

def load_checkpoint(config, model):
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    max_accuracy = [1e6] * 3
    print('Load ckpt from:', config.MODEL.RESUME)
    return max_accuracy

def save_checkpoint(config, epoch, model, max_accuracy):
    save_state = {'model': model.state_dict(), 'max_accuracy': max_accuracy, 'config': config}
    save_path = os.path.join(config.OUTPUT_DIR, f'ckpt_epoch_{epoch}.pth')
    torch.save(save_state, save_path)

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"Found the latest ckpt: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

class CurvePlotter:
    def __init__(self, label, savedir):
        self.label = label
        self.epo = []
        self.data = []
        self.key_epoch = None
        self.savepath = os.path.join(savedir, f'{label}_curve.png')

    def plot_curve(self):
        if not self.epo or not self.data:
            raise ValueError("Both 'epo' and 'data' lists must have values to plot.")
        fig = plt.figure()
        plt.title(self.label)
        plt.plot(self.epo, self.data, label='Data Curve')
        if self.key_epoch is not None and self.key_epoch in self.epo:
            key_index = self.epo.index(self.key_epoch)
            plt.plot(self.epo[key_index], self.data[key_index], 'ro', label='best epoch')
            plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(self.label)
        plt.grid(True)
        plt.savefig(self.savepath)
        plt.close(fig)

    def add(self, epoch, value):
        self.epo.append(epoch)
        self.data.append(value)

    def set_key_epoch(self, epoch):
        self.key_epoch = epoch

    def add_and_plot(self, epoch, value):
        self.add(epoch, value)
        self.plot_curve()

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
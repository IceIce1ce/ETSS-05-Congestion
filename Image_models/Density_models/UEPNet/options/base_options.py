import argparse
import models
import dataset
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # general config
        parser.add_argument('--type_dataset', type=str, default='sha')
        parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A')
        parser.add_argument('--ckpt_dir', type=str, default='saved_sha')
        parser.add_argument('--dataset_mode', type=str, default='shtechparta')
        parser.add_argument('--gpu_ids', type=str, default='0')
        parser.add_argument('--train_lists', type=str, default='datasets/ShanghaiTech/part_A/train_sha.txt')
        parser.add_argument('--test_lists', type=str, default='datasets/ShanghaiTech/part_A/test_sha.txt')
        # model config
        parser.add_argument('--model', type=str, default='uep', help='chooses which model to use. [uep | ]')
        # testing config
        parser.add_argument('--num_threads', default=8, type=int)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--crop_ratio', type=float, default=0.5)
        parser.add_argument('--gaussian_kernel_size', type=float, default=15)
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"))
        parser.add_argument('--epoch', type=str, default='latest')
        parser.add_argument('--load_iter', type=int, default='0')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--pad_batch', action='store_true')
        parser.add_argument('--suffix', default='', type=str)
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)
        opt, _ = parser.parse_known_args()
        dataset_name = opt.dataset_mode
        dataset_option_setter = dataset.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt
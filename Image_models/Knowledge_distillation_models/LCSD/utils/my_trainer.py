import os
from models.scale_regression import ScalenetTrainer
from utils.data_generate import DatasetGenerator
from models.counter.trainer import CounterTrainer
from models.detector.trainer import DetectorTrainer
import utils.log_utils as log_utils

class MyTrainer:
    def __init__(self, args):
        self.args = args
    
    def setup(self):
        args = self.args
        if args.type_dataset.lower() == 'cityuhk-x':
            args.img_height = 384
            args.img_width = 512
            args.crop_size = 256
        elif args.type_dataset.lower() == 'mall':
            args.img_height = 480
            args.img_width = 640
            args.crop_size = 256
        elif args.type_dataset.lower() == 'ucsd':
            args.img_height = 158
            args.img_width = 238
            args.crop_size = 128
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        self.logger = log_utils.get_logger(os.path.join(args.output_dir, 'train.log'))
        args.scene_dataset = os.path.join(args.input_dir, args.type_dataset, args.scene, args.scene_dataset)
        self.data_generator = DatasetGenerator(args)
        self.counter = CounterTrainer(args, self.logger)
        self.detector = DetectorTrainer(args, self.logger)
        self.scale_net = ScalenetTrainer(args)

    def train(self):
        pre_dis = None
        pre_num = None
        scale_model = None
        for i in range(1, self.args.iterative_num + 1):
            print('Epoch: [{}/{}]'.format(i, self.args.iterative_num + 1))
            # generate a synthetic dataset
            print('Generating synthetic dataset for counter model')
            com_data, base_image = self.data_generator.generate(i, pre_dis, pre_num, scale_model)
            # train counter model
            print('Training counter model')
            self.counter.train(com_data, scale_model, i)
            self.counter.save_model(i)
            # train detector model
            print('Generating synthetic dataset for detector model')
            com_data, _ = self.data_generator.generate(i, pre_dis, pre_num)
            print('Training detector model')
            self.detector.train(com_data)
            self.detector.save_model(i)
            _, scale_data = self.detector.predict(i)
            # train scale_net model
            print('Training scale_net model')
            self.scale_net.linear_fit(scale_data, i)
            scale_model = self.scale_net
            pre_dis, pre_num = self.counter.predict(i, base_image)
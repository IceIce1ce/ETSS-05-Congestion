import os
import logging
from utils.logger import setlogger

class Trainer(object):
    def __init__(self, args):
        self.save_dir = args.output_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'train.log'))
        for k, v in args.__dict__.items():
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        pass

    def train(self):
        pass
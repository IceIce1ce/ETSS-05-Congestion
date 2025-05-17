import argparse
import random
from engine import Trainer, evaluate_crowd_counting
import numpy as np
import os
import shutil
import torch
from datasets import build_dataset
from models import build_model
from config import cfg, merge_from_file, merge_from_list
import warnings
warnings.filterwarnings('ignore')

def train(cfg):
    output_dir = cfg.OUTPUT_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(cfg.config_file, output_dir)
    # train and test loader
    train_dl, val_dl = build_dataset(cfg=cfg)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # model
    model, criterion = build_model(cfg=cfg, training=True)
    model.cuda()
    criterion.cuda()
    # train
    trainer = Trainer(cfg, model, train_dl, val_dl, criterion)
    for epoch in range(trainer.train_epoch, trainer.epochs):
        for batch in trainer.train_dl:
            trainer.step(batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()

def test(cfg):
    source_dir = cfg.OUTPUT_DIR
    output_dir = os.path.join(source_dir, "%s_%.2f"%(cfg.DATASETS.DATASET, cfg.TEST.THRESHOLD))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vis_val_path = None
    if cfg.VIS:  
        vis_val_path = os.path.join(output_dir, 'sample_result_for_val/')
        if not os.path.exists(vis_val_path):
            os.makedirs(vis_val_path)
    # train and test loader
    train_dl, val_dl = build_dataset(cfg=cfg)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # model
    model = build_model(cfg=cfg, training=False)
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params: %d' % n_parameters)
    pretrained_dict = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)
    # testing
    result = evaluate_crowd_counting(model, val_dl, next(model.parameters()).device, cfg.TEST.THRESHOLD, vis_val_path)
    print('MAE: %.4f, MSE: %.4f ' % (result[0], result[1]))

def main(cfg):
    seed = cfg.SEED
    if seed != None:
        g = torch.Generator()
        g.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
    if cfg.test:
        test(cfg)
    else:
        train(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="configs/SHHA_basic.yml")
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg = merge_from_file(cfg, args.config_file)
    cfg = merge_from_list(cfg, args.opts)
    cfg.config_file = args.config_file
    cfg.test = args.test
    if cfg.test == False:
        print('Training dataset:', cfg.DATASETS.DATASET)
    else:
        print('Testing dataset:', cfg.DATASETS.DATASET)
    main(cfg)
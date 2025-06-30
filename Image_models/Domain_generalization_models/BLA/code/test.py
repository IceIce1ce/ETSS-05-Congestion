import os
import sys
sys.path.append(os.path.abspath((__file__)))
from options_test import global_variables as opt_conf
from tools.terminal_log import create_log_file_terminal, save_opt, create_exp_dir
from tools.log import AverageMeter
from tools.progress.bar import Bar
import torch
import time
import random
import glob
import warnings
warnings.filterwarnings("ignore")

def test(epoch, test_vision, model, test_loader, log):
    with torch.no_grad():
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        bar = Bar('Testing', max=len(test_loader))
        end = time.time()
        for step, data in enumerate(test_loader):
            data_time.update(time.time() - end)
            end = time.time()
            res = model.inference(data, epoch, step, test_vision)
            batch_time.update(time.time() - end)
            end = time.time()
            str_plus=''
            for k,v in res.items():
                str_plus+=' | {key:}:{value:.4f}'.format(key=k,value=v)
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(batch=step + 1, size=len(test_loader), data=data_time.val, bt=batch_time.val, total=bar.elapsed_td, eta=bar.eta_td,) + str_plus
            bar.next()
        bar.finish()
    return res

if __name__ == '__main__':
    option = dict([arg.split('=') for arg in sys.argv[1:]])
    opt = opt_conf(**option)
    log_output = create_log_file_terminal(opt.log_txt_path)
    save_opt(opt,opt.log_txt_path)
    scripts_to_save = glob.glob('code/*')
    create_exp_dir(opt.log_root_path, scripts_to_save)
    from importlib import import_module
    # model
    net = import_module('models.{}'.format(opt.model))
    exec('model=net.{}(opt)'.format(opt.model))
    model = model.cuda()
    # test loader
    dataloader = import_module('datasets.{}'.format(opt.tar_dataset))
    exec('test_loader_tar,test_data_tar=dataloader.{}(opt)'.format(opt.tar_dataset))
    sample_num = min(len(test_loader_tar), opt.vision_each_epoch) # 182
    test_vision_tar = random.sample(list(range(len(test_loader_tar))), sample_num)
    assert opt.model_for_load
    checkpoint = torch.load(opt.model_for_load)
    model.load_state_dict(checkpoint['net'])
    model.next_epoch()
    results = test(0, test_vision_tar, model, test_loader_tar, log_output)
    log_output.info('Results in {}'.format(opt.tar_dataset))
    for k,v in results.items():
        log_output.info('{}: {}'.format(k,v))
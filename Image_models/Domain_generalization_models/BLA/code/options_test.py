import time
import os

class global_variables(object):
    num_gpu = 1
    local_rank = 0
    iter_each_train_epoch=0
    seed = 1234
    time_= time.localtime()
    time_= time.strftime('%Y-%m-%d-%H-%M',time_)
    log_root_path=''
    log_txt_path=''
    comment = ''
    dataroot_SHA = '/opt/data/common/shanghaitechA'
    tar_dataset = 'SHA_test'
    img_interpolate_mode = 'bilinear'# 'nearest' 'bicubic'
    mean_std = ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    gt_factor = 100.0
    sigma = 4
    kernel_size = 15
    num_workers = 8
    test_batch_size = 1
    model = ''
    model_for_load = ''
    split_den_level = [0,0.005,10000]
    model_scale_factor = 8
    gradient_scalar = -0.01
    vision_each_epoch=5000

    def __init__(self, **kwself):
        for k, v in kwself.items():
            if k == '--local_rank':
                k = 'local_rank'
            if not hasattr(self, k):
                print("Warning: opt has not attribut {}".format(k))
                import pdb
                pdb.set_trace()
                self.__dict__.update({k: v})
            tp = eval('type(self.{0})'.format(k))
            if tp == type(''):
                setattr(self, k, tp(v))
            elif tp == type([]):
                tp=eval('type(self.{0}[0])'.format(k))
                if tp==type('1'):
                    v=v[1:-1].split(',')
                    setattr(self, k, v)
                else:
                    setattr(self, k, eval(v))
            else:
                setattr(self, k, eval(v))
        if self.comment:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}_{}'.format(self.time_,self.model,self.comment))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}_{}.txt'.format(self.time_,self.model,self.comment))
        else:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}'.format(self.time_,self.model))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}.txt'.format(self.time_,self.model))
        assert self.test_batch_size==1
        os.makedirs(self.log_root_path)
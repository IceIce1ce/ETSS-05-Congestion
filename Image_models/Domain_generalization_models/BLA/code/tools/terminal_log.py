import logging
import os
import shutil

def create_log_file_terminal(logfile_path,log_name='log_name'):#.txt
    log = logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s-%(name)-12s: terminal %(levelname)-8s %(message)s')
    console.setFormatter(console_formatter)
    log.addHandler(console)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')   
    file=logging.FileHandler(logfile_path) 
    file.setFormatter(file_formatter)
    log.addHandler(file)
    return log

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    if scripts_to_save is not None:
        if os.path.exists(os.path.join(path, 'scripts')):
            shutil.rmtree(os.path.join(path, 'scripts'))
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            if os.path.isdir(script):
                shutil.copytree(script, dst_file)
            else:
                shutil.copyfile(script, dst_file)

def save_opt(opt,log_path):
    if '.txt' in log_path:
        txt_save_path=log_path
    else:
        txt_save_path=os.path.join(log_path,'opt_save.txt')
    if not os.path.exists(os.path.dirname(txt_save_path)):
        os.makedirs(os.path.dirname(txt_save_path))
    with open(txt_save_path,'w') as f:
        for k, v in opt.__class__.__dict__.items():
            if not k.startswith('__'):
                f.write('{0} \t {1} \n'.format(k, getattr(opt, k)))
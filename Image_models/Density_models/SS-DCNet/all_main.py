import os
import argparse
from main_process import main
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', default='SHB', choices=['SHA','SHB','QNRF'])
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    dataset_idxlist = {'SHA': 0, 'SHB': 1, 'QNRF': 2}
    dataset_list = ['SH_partA', 'SH_partB', 'UCF-QNRF_ECCV18']
    dataset_max = [[22], [7], [8]]
    dataset_choose = [dataset_idxlist[args.type_dataset]]
    for di in dataset_choose:
        opt = dict()
        opt['dataset'] = dataset_list[di]
        opt['max_list'] = dataset_max[di]
        opt['root_dir'] = os.path.join(r'data',opt['dataset'])
        opt['num_workers'] = 0
        opt['IF_savemem_train'] = False
        opt['IF_savemem_test'] = False
        opt['test_batch_size'] = 1
        opt['psize'],opt['pstride'] = 64,64
        opt['div_times'] = 2
        parse_method_dict = {0:'maxp'}
        opt['parse_method'] = parse_method_dict[0]
        opt['max_num'] = opt['max_list'][0]
        opt['partition'] = 'two_linear'
        opt['step'] = 0.5
        opt['model_path'] = os.path.join('model', args.type_dataset)
        main(opt)
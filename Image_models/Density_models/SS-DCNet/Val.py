import torch
from time import time
import math
from IOtools import txt_write 

def test_phase(opt,net,testloader,log_save_path=None):
    with torch.no_grad():
        net.eval()
        start = time()
        avg_frame_rate = 0
        mae = 0.0
        rmse = 0.0
        me = 0.0
        for j, data in enumerate(testloader):
            inputs , labels = data['image'], data['all_num'] # [1, 3, 704, 1024], [1, 1, 1]
            inputs,labels = inputs.type(torch.float32),labels.unsqueeze(1).type(torch.float32)
            inputs, labels = inputs.cuda(), labels.cuda()
            features = net(inputs)
            div_res = net.resample(features)
            merge_res = net.parse_merge(div_res)
            outputs = merge_res['div' + str(net.div_times)] # [1, 1, 44, 64]
            del merge_res
            pre = outputs.sum()
            gt = labels.sum()
            mae += abs(pre-gt)
            rmse += (pre-gt)*(pre-gt)
            me += (pre-gt)
            end = time()
            running_frame_rate = opt['test_batch_size'] * float( 1 / (end - start))
            avg_frame_rate = (avg_frame_rate*j + running_frame_rate)/(j+1)
            if j % 1 == 0:
                print('Test:[%5d/%5d] pre: %.3f gt:%.3f err:%.3f frame: %.2fHz/%.2fHz' % (j + 1,len(testloader), pre, gt,pre-gt, running_frame_rate,avg_frame_rate))
                start = time()
        im_num = len(testloader)
        log_str = '%10s\t %8s\t &%8s\t &%8s\t\\\\' % (' ','mae','rmse','me')+'\n'
        log_str += '%-10s\t %8.3f\t %8.3f\t %8.3f\t' % ( 'test',mae/im_num,math.sqrt(rmse/im_num),me/im_num ) + '\n'
        print(log_str)
        if log_save_path:
            txt_write(log_save_path,log_str,mode='w')
    im_num = len(testloader)
    test_dict=dict()
    test_dict['mae'] = mae / im_num
    test_dict['mse'] = math.sqrt(rmse/(im_num))
    test_dict['me'] = me/(im_num)
    return test_dict
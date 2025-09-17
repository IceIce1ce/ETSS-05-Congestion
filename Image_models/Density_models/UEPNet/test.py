import warnings
warnings.filterwarnings("ignore")
import numpy as np
from options.train_options import TrainOptions
from dataset import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    print('Testing dataset:', opt.type_dataset)
    # test loader
    dataset = create_dataset(opt)
    # model
    model = create_model(opt)
    model.setup(opt)
    model.load_networks('best')
    model.to_eval()
    if opt.test_lists:
        if len(opt.test_lists.strip().split(',')) > 0:
            process_func = dataset.get_preprocess_func_for_test()
            for test_list in opt.test_lists.strip().split(','):
                eval_img_list = dataset.get_test_imgs(test_list)
                all_imgs = list(eval_img_list.keys())
                pred_gt_array = np.zeros((len(all_imgs), 2)) # [182, 2]
                for img_ind, img_path in enumerate(all_imgs):
                    model.set_input(process_func(img_path))
                    pred_gt_array[img_ind, 0] = model.predict()
                    txtr = img_path[:-3] + 'txt'
                    txtfile = open(txtr)
                    gts = txtfile.readlines()
                    gt = len(gts)
                    txtfile.close()
                    pred_gt_array[img_ind, 1] = gt
                mae = np.mean(np.abs(pred_gt_array[:, 0] - pred_gt_array[:, 1]))
                mse = np.sqrt(np.mean(np.power(pred_gt_array[:, 0] - pred_gt_array[:, 1], 2)))
                print('File name: {}, MAE: {:.2f}, MSE: {:.2f}'.format(test_list, mae, mse))
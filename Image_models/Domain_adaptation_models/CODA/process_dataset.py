import os
import argparse

def prepareShangHaiTech(args):
    for part in ['part_A_final','part_B_final']:
        for phase in ['train_data','test_data']:
            DATASET_PATH = os.path.join(args.input_dir, part, phase)
            fout = open(DATASET_PATH + '.txt', 'w+')
            for img_name in os.listdir(os.path.join(DATASET_PATH, 'images')):
                image_path = os.path.join(DATASET_PATH, 'images', img_name)
                gt_path = os.path.join(DATASET_PATH, 'ground_truth', 'GT_' + img_name.split('.')[0] + '.mat')
                fout.write(image_path + ' ' + gt_path + '\n')
            fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='datasets/sha')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    if args.type_dataset == 'sha':
        prepareShangHaiTech(args)
    else:
        print('This dataset does not exist')
        raise NotImplementedError
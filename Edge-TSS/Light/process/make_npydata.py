import os
import numpy as np
import argparse

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # process sha dataset
    try:
        shanghaiAtrain_path = os.path.join(args.sh_dir, 'part_A_final/train_data/images/')
        shanghaiAtest_path = os.path.join(args.sh_dir, 'part_A_final/test_data/images/')
        train_list = []
        for filename in os.listdir(shanghaiAtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(shanghaiAtrain_path + filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiA_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(shanghaiAtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(shanghaiAtest_path + filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiA_test.npy'), test_list)
        print("Generate SHA image list successfully")
    except:
        print("The SHA dataset path is wrong. Please check you path")
    # process shb dataset
    try:
        shanghaiBtrain_path = os.path.join(args.sh_dir, 'part_B_final/train_data/images/')
        shanghaiBtest_path = os.path.join(args.sh_dir, 'part_B_final/test_data/images/')
        train_list = []
        for filename in os.listdir(shanghaiBtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(shanghaiBtrain_path + filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiB_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(shanghaiBtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(shanghaiBtest_path + filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiB_test.npy'), test_list)
        print("Generate SHB image list successfully")
    except:
        print("The SHB dataset path is wrong. Please check your path")
    # process qnrf dataset
    try:
        Qnrf_train_path = os.path.join(args.qnrf_dir, 'train_data/images/')
        Qnrf_test_path = os.path.join(args.qnrf_dir, 'test_data/images/')
        train_list = []
        for filename in os.listdir(Qnrf_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(Qnrf_train_path + filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'qnrf_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(Qnrf_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(Qnrf_test_path + filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'qnrf_test.npy'), test_list)
        print("Generate QNRF image list successfully")
    except:
        print("The QNRF dataset path is wrong. Please check your path")
    # proces jhu dataset
    try:
        Jhu_train_path = os.path.join(args.jhu_dir, 'train/images_2048/')
        Jhu_val_path = os.path.join(args.jhu_dir, 'val/images_2048/')
        jhu_test_path = os.path.join(args.jhu_dir, 'test/images_2048/')
        train_list = []
        for filename in os.listdir(Jhu_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(Jhu_train_path + filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_train.npy'), train_list)
        val_list = []
        for filename in os.listdir(Jhu_val_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(Jhu_val_path + filename)
        val_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_val.npy'), val_list)
        test_list = []
        for filename in os.listdir(jhu_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(jhu_test_path + filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_test.npy'), test_list)
        print("Generate JHU image list successfully")
    except:
        print("The JHU dataset path is wrong. Please check your path")
    # process NWPU dataset
    try:
        f = open(os.path.join(args.nwpu_dir, "train.txt"), "r")
        train_list = f.readlines()
        f = open(os.path.join(args.nwpu_dir, "val.txt"), "r")
        val_list = f.readlines()
        f = open(os.path.join(args.nwpu_dir, "test.txt"), "r")
        test_list = f.readlines()
        root = os.path.join(args.nwpu_dir, 'images_2048/')
        train_img_list = []
        for i in range(len(train_list)):
            fname = train_list[i].split(' ')[0] + '.jpg'
            train_img_list.append(root + fname)
        np.save(os.path.join(args.output_dir, 'nwpu_train.npy'), train_img_list)
        val_img_list = []
        for i in range(len(val_list)):
            fname = val_list[i].split(' ')[0] + '.jpg'
            val_img_list.append(root + fname)
        np.save(os.path.join(args.output_dir, 'nwpu_val.npy'), val_img_list)
        test_img_list = []
        root = root.replace('images', 'test_data')
        for i in range(len(test_list)):
            fname = test_list[i].split(' ')[0] + '.jpg'
            test_img_list.append(root + fname)
        np.save(os.path.join(args.output_dir, 'nwpu_test.npy'), test_img_list)
        print("Generate NWPU image list successfully")
    except:
        print("The NWPU dataset path is wrong. Please check your path")
    # process trancos dataset
    try:
        trancos_train_path = os.path.join(args.trancos_dir, 'train_data/images/')
        trancos_test_path = os.path.join(args.trancos_dir, 'test_data/images/')
        train_list = []
        for filename in os.listdir(trancos_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(trancos_train_path + filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'trancos_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(trancos_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(trancos_test_path + filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'trancos_test.npy'), test_list)
        print("Generate TRANCOS image list successfully")
    except:
        print("The TRANCOS dataset path is wrong. Please check you path")
    # process suwon dataset
    try:
        suwon_train_path = os.path.join(args.suwon_dir, 'train_data/images/')
        suwon_test_path = os.path.join(args.suwon_dir, 'test_data/images/')
        train_list = []
        for filename in os.listdir(suwon_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(suwon_train_path + filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'suwon_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(suwon_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(suwon_test_path + filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'suwon_test.npy'), test_list)
        print("Generate Suwon image list successfully")
    except:
        print("The Suwon dataset path is wrong. Please check you path")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sh_dir', type=str, default='datasets/ShanghaiTech')
    parser.add_argument('--qnrf_dir', type=str, default='datasets/UCF-QNRF')
    parser.add_argument('--jhu_dir', type=str, default='datasets/jhu_crowd_v2.0')
    parser.add_argument('--nwpu_dir', type=str, default='datasets/NWPU_CLTR')
    parser.add_argument('--trancos_dir', type=str, default='datasets/TRANCOS')
    parser.add_argument('--suwon_dir', type=str, default='datasets/Suwon')
    parser.add_argument('--output_dir', type=str, default='npydata')
    args = parser.parse_args()
    main(args)
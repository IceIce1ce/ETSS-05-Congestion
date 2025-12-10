import os
import numpy as np
import argparse

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    try:
        Jhu_train_path = os.path.join(args.jhu_path, 'train/images_2048')
        Jhu_val_path = os.path.join(args.jhu_path, 'val/images_2048')
        jhu_test_path = os.path.join(args.jhu_path, 'test/images_2048')
        train_list = []
        for filename in os.listdir(Jhu_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(Jhu_train_path, filename))
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_train.npy'), train_list)
        val_list = []
        for filename in os.listdir(Jhu_val_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(os.path.join(Jhu_val_path, filename))
        val_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_val.npy'), val_list)
        test_list = []
        for filename in os.listdir(jhu_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(jhu_test_path, filename))
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_test.npy'), test_list)
    except:
        print("The JHU dataset path is wrong")
    try:
        f = open("data/NWPU_list/train.txt", "r")
        train_list = f.readlines()
        f = open("data/NWPU_list/val.txt", "r")
        val_list = f.readlines()
        root = os.path.join(args.nwpu_path, 'gt_detr_map')
        if not os.path.exists(root):
            print("The NWPU dataset path is wrong")
        else:
            train_img_list = []
            for i in range(len(train_list)):
                fname = train_list[i].split(' ')[0] + '.jpg'
                train_img_list.append(os.path.join(root, fname))
            val_img_list = []
            for i in range(len(val_list)):
                fname = val_list[i].split(' ')[0] + '.jpg'
                val_img_list.append(os.path.join(root, fname))
            np.save(os.path.join(args.output_dir, 'nwpu_train.npy'), train_img_list)
            np.save(os.path.join(args.output_dir, 'nwpu_val.npy'), val_img_list)
    except:
        print("The NWPU dataset path is wrong")
    try:
        trancos_train_path = os.path.join(args.trancos_path, 'train_data/images')
        trancos_test_path = os.path.join(args.trancos_path, 'test_data/images')
        train_list = []
        for filename in os.listdir(trancos_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(trancos_train_path, filename))
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'trancos_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(trancos_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(trancos_test_path, filename))
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'trancos_test.npy'), test_list)
    except:
        print("The TRANCOS dataset path is wrong")
    try:
        drone_vehicle_train_path = os.path.join(args.drone_vehicle_path, 'train_data/images')
        drone_vehicle_test_path = os.path.join(args.drone_vehicle_path, 'test_data/images')
        train_list = []
        for filename in os.listdir(drone_vehicle_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(drone_vehicle_train_path, filename))
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'drone_vehicle_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(drone_vehicle_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(drone_vehicle_test_path, filename))
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'drone_vehicle_test.npy'), test_list)
    except:
        print("The DroneVehicle dataset path is wrong")
    try:
        suwon_train_path = os.path.join(args.suwon_path, 'train_data/images')
        suwon_test_path = os.path.join(args.suwon_path, 'test_data/images')
        train_list = []
        for filename in os.listdir(suwon_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(suwon_train_path, filename))
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'suwon_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(suwon_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(suwon_test_path, filename))
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'suwon_test.npy'), test_list)
    except:
        print("The Suwon dataset path is wrong")
    try:
        sub_suwon_train_path = os.path.join(args.sub_suwon_path, 'train_data/images')
        sub_suwon_test_path = os.path.join(args.sub_suwon_path, 'test_data/images')
        train_list = []
        for filename in os.listdir(sub_suwon_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(sub_suwon_train_path, filename))
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'sub_suwon_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(sub_suwon_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(sub_suwon_test_path, filename))
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'sub_suwon_test.npy'), test_list)
    except:
        print("The Sub Suwon dataset path is wrong")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jhu_path', type=str, default='data/jhu_crowd_v2.0')
    parser.add_argument('--nwpu_path', type=str, default='data/NWPU_CLTR')
    parser.add_argument('--trancos_path', type=str, default='data/TRANCOS')
    parser.add_argument('--drone_vehicle_path', type=str, default='data/DroneVehicle')
    parser.add_argument('--suwon_path', type=str, default='data/Suwon')
    parser.add_argument('--sub_suwon_path', type=str, default='data/Sub_Suwon')
    parser.add_argument('--output_dir', type=str, default='npydata')
    args = parser.parse_args()
    main(args)
import json
import os
import cv2
from scipy import io as sio
from glob import glob
from tqdm import tqdm
import argparse

def main(args):
    sub_dirs = ["train_data", "test_data"]
    categories = [{"id": 0, "name": "person", "supercategory": "person"}]
    for sub_dir in sub_dirs:
        img_file_list = sorted(glob(f"{args.input_dir}/{sub_dir}/images/**.jpg"))
        mat_file_list = sorted(glob(f"{args.input_dir}/{sub_dir}/ground_truth/*.mat"))
        output_file = f"{args.input_dir}/{sub_dir}_annotation.json"
        images = []
        annotations = []
        anno_id = 0
        idx = 0
        for img_path, mat_path in tqdm(zip(img_file_list, mat_file_list)):
            img = cv2.imread(img_path)
            mat = sio.loadmat(mat_path)
            file_name = os.path.basename(img_path)
            idx += 1
            points = mat["image_info"][0][0][0][0][0]
            obj = {}
            obj["bbox"] = []
            obj["iscrowd"] = []
            for pt in points:
                obj["bbox"].append((pt[0], pt[1], 1, 1))
            images.append({"id": int(idx), "file_name": file_name, "width": img.shape[0], "height": img.shape[1]})
            for box in obj["bbox"]:
                annotations.append({"id": anno_id, "image_id": int(idx), "bbox":[box[0], box[1], box[2], box[3]], "iscrowd": 0, "category_id": 0, "area": (box[2]) * (box[3])})
                anno_id += 1
        with open(output_file,"w") as f:
            json.dump({"annotations": annotations, "images": images, "categories": categories}, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)
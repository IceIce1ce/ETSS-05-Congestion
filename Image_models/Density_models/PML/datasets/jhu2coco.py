import json
import os
import cv2
from tqdm import tqdm
import argparse

def main(args):
    sub_dirs = ["train", "val", "test"]
    for sub_dir in sub_dirs:
        idx_file = f"{args.input_dir}/{sub_dir}/image_labels.txt"
        output_file = f"{args.input_dir}/{sub_dir}_annotation.json"
        images = []
        annotations = []
        anno_id = 0
        categories = [{"id": 0, "name": "person", "supercategory": "person"}]
        with open(idx_file, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip("\n").split(",")
                idx, cnt, scence_label, weather_label = line[0], line[1], line[2], line[3]
                file_name = f"{idx}.jpg"
                img_path = os.path.join(args.input_dir, sub_dir, "images", f"{idx}.jpg")
                gt_path = os.path.join(args.input_dir, sub_dir, "gt", f"{idx}.txt")
                img = cv2.imread(img_path)
                obj = {}
                with open(gt_path,"r") as f_gt:
                    gt_lines = f_gt.readlines()
                    obj["bbox"] = []
                    obj["iscrowd"] = []
                    for line in gt_lines:
                        obj["bbox"].append(list(map(float,line.strip("\n").split(" ")[:4])))
                        obj["iscrowd"].append(not line.strip("\n").split(" ")[4]=="1")
                images.append({"id": int(idx), "file_name": file_name, "width": img.shape[0], "height": img.shape[1], "scence": scence_label, "weather": weather_label, "cnt": cnt})
                for is_crowd, box in zip(obj["iscrowd"], obj["bbox"]):
                    annotations.append({"id": anno_id, "image_id": int(idx), "bbox": [box[0] - box[2] * 0.5, box[1] - box[3] * 0.5, box[2], box[3]], "iscrowd": is_crowd,
                                        "category_id": 0, "area": (box[2]) * (box[3])})
                    anno_id += 1
        with open(output_file, "w") as f:
            json.dump({"annotations": annotations, "images": images, "categories": categories}, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='jhu')
    parser.add_argument('--input_dir', type=str, default='data/jhu_crowd_v2.0')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)
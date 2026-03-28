import argparse
import json
import os
import cv2
from tqdm import tqdm

def main(args):
    idx_file = os.path.join(args.input_dir, f"{args.mode}.txt")
    output_file = os.path.join(args.input_dir, f"{args.mode}_annotation.json")
    images = []
    annotations = []
    anno_id = 0
    categories = [{"id": 0, "name": "person", "supercategory": "person"}]
    with open(idx_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip("\n").split(" ")
            idx, bright_label, scence_label = line[0], line[1], line[2]
            img_path = os.path.join(args.input_dir, "images", f"{idx}.jpg")
            json_path = os.path.join(args.input_dir, "jsons", f"{idx}.json")
            img = cv2.imread(img_path) # [2448, 3264, 32]
            with open(json_path, "r") as f_json:
                obj = json.load(f_json)
                boxes = obj["boxes"]
                file_name = obj["img_id"]
            images.append({"id": int(idx), "file_name": file_name, "width": img.shape[0], "height": img.shape[1], "scence": scence_label, "brightness": bright_label})
            for box in boxes:
                annotations.append({"id": anno_id, "image_id": int(idx), "bbox": [box[0], box[1], (box[2] - box[0]), (box[3] - box[1])], "iscrowd": 0, "category_id": 0,
                                    "area": (box[2] - box[0]) * (box[3] - box[1])})
                anno_id += 1
    with open(output_file, "w") as f:
        json.dump({"annotations": annotations, "images": images, "categories": categories}, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='nwpu')
    parser.add_argument('--input_dir', type=str, default='data/NWPU-Crowd')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)
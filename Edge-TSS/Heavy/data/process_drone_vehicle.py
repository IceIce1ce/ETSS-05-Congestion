import os
import xml.etree.ElementTree as ET
import argparse
import numpy as np
from scipy.io import savemat
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

def polygon_center(x_coords, y_coords):
    x_center = sum(x_coords) / len(x_coords)
    y_center = sum(y_coords) / len(y_coords)
    return int(round(x_center)), int(round(y_center))

# get both center and non-center
# def process_split(split_dir):
#     labels_dir = os.path.join(split_dir, "labels")
#     txt_dir = os.path.join(split_dir, "txt")
#     os.makedirs(txt_dir, exist_ok=True)
#     xml_files = [f for f in os.listdir(labels_dir) if f.endswith(".xml")]
#     for xml_file in xml_files:
#         xml_path = os.path.join(labels_dir, xml_file)
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         centers = []
#         for obj in root.findall("object"):
#             polygon = obj.find("polygon")
#             if polygon is not None:
#                 x_coords = [int(polygon.find(f"x{i + 1}").text) for i in range(4)]
#                 y_coords = [int(polygon.find(f"y{i + 1}").text) for i in range(4)]
#                 x_center, y_center = polygon_center(x_coords, y_coords)
#                 centers.append(f"{x_center}\t{y_center}")
#         txt_filename = os.path.splitext(xml_file)[0] + ".txt"
#         txt_path = os.path.join(txt_dir, txt_filename)
#         with open(txt_path, "w") as f:
#             f.write("\n".join(centers))
#         print(f"Processed {xml_file} to {txt_filename} ({len(centers)} centers)")

# remove non-center
# def process_split(split_dir):
#     labels_dir = os.path.join(split_dir, "labels")
#     images_dir = os.path.join(split_dir, "images")
#     txt_dir = os.path.join(split_dir, "txt")
#     os.makedirs(txt_dir, exist_ok=True)
#     xml_files = [f for f in os.listdir(labels_dir) if f.endswith(".xml")]
#     for xml_file in xml_files:
#         xml_path = os.path.join(labels_dir, xml_file)
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         centers = []
#         for obj in root.findall("object"):
#             polygon = obj.find("polygon")
#             if polygon is not None:
#                 x_coords = [int(polygon.find(f"x{i + 1}").text) for i in range(4)]
#                 y_coords = [int(polygon.find(f"y{i + 1}").text) for i in range(4)]
#                 x_center, y_center = polygon_center(x_coords, y_coords)
#                 centers.append(f"{x_center}\t{y_center}")
#         txt_filename = os.path.splitext(xml_file)[0] + ".txt"
#         txt_path = os.path.join(txt_dir, txt_filename)
#         if centers:
#             with open(txt_path, "w") as f:
#                 f.write("\n".join(centers))
#             print(f"Processed {xml_file} to {txt_filename} ({len(centers)} centers)")
#         else:
#             image_file = os.path.splitext(xml_file)[0] + ".jpg"
#             image_path = os.path.join(images_dir, image_file)
#             if os.path.exists(image_path):
#                 os.remove(image_path)
#                 print(f"Deleted image: {image_file}")
#             os.remove(xml_path)
#             print(f"Deleted XML: {xml_file}")

# remove number of center <= 20
def process_split(split_dir, min_centers=20):
    labels_dir = os.path.join(split_dir, "labels")
    images_dir = os.path.join(split_dir, "images")
    txt_dir = os.path.join(split_dir, "txt")
    os.makedirs(txt_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(labels_dir) if f.endswith(".xml")]
    for xml_file in xml_files:
        xml_path = os.path.join(labels_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        centers = []
        for obj in root.findall("object"):
            polygon = obj.find("polygon")
            if polygon is not None:
                x_coords = [int(polygon.find(f"x{i + 1}").text) for i in range(4)]
                y_coords = [int(polygon.find(f"y{i + 1}").text) for i in range(4)]
                x_center, y_center = polygon_center(x_coords, y_coords)
                centers.append(f"{x_center}\t{y_center}")
        if len(centers) > min_centers:
            txt_filename = os.path.splitext(xml_file)[0] + ".txt"
            txt_path = os.path.join(txt_dir, txt_filename)
            with open(txt_path, "w") as f:
                f.write("\n".join(centers))
            print(f"Processed {xml_file} to {txt_filename} ({len(centers)} centers)")
        else:
            image_file_jpg = os.path.splitext(xml_file)[0] + ".jpg"
            image_file_png = os.path.splitext(xml_file)[0] + ".png"
            image_path_jpg = os.path.join(images_dir, image_file_jpg)
            image_path_png = os.path.join(images_dir, image_file_png)
            for image_path in [image_path_jpg, image_path_png]:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Deleted image: {os.path.basename(image_path)}")
            os.remove(xml_path)
            print(f"Deleted XML: {xml_file} (only {len(centers)} centers)")

def convert_txt_to_mat(txt_path, mat_path):
    coordinates = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    coordinates.append([x, y])
    coords_array = np.array(coordinates, dtype=np.float64)
    num_points = len(coordinates)
    location_data = coords_array
    number_data = np.array([[num_points]], dtype=np.uint8)
    structured_array = np.array([[(location_data, number_data)]], dtype=[('location', 'O'), ('number', 'O')])
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = structured_array
    savemat(mat_path, {'image_info': image_info, '__header__': b'MATLAB 5.0 MAT-file, Created by Python', '__version__': '1.0', '__globals__': []})
    return num_points

def process_dataset_folder(data_folder):
    txt_folder = os.path.join(data_folder, 'txt')
    gt_folder = os.path.join(data_folder, 'ground_truth')
    if not os.path.exists(txt_folder):
        print(f"Warning: {txt_folder} does not exist. Skipping...")
        return
    os.makedirs(gt_folder, exist_ok=True)
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"Warning: No .txt files found in {txt_folder}")
        return
    for txt_file in sorted(txt_files):
        txt_path = os.path.join(txt_folder, txt_file)
        mat_file = txt_file.replace('.txt', '.mat')
        mat_path = os.path.join(gt_folder, mat_file)
        try:
            num_points = convert_txt_to_mat(txt_path, mat_path)
            print(f"Processed {txt_file} to {mat_file} ({num_points} points)")
        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")

def main_process(args):
    # process train part
    train_folder = os.path.join(args.input_dir, 'train_data')
    if os.path.exists(train_folder):
        process_dataset_folder(train_folder)
    else:
        print(f"Warning: {train_folder} not found")
    # process test part
    test_folder = os.path.join(args.input_dir, 'test_data')
    if os.path.exists(test_folder):
        process_dataset_folder(test_folder)
    else:
        print(f"Warning: {test_folder} not found")

def main_clean(split_dir):
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith(".xml")}
    matched_files = image_files & label_files
    for img_name in image_files - matched_files:
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(images_dir, img_name + ext)
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted unmatched image: {img_path}")
    for lbl_name in label_files - matched_files:
        lbl_path = os.path.join(labels_dir, lbl_name + ".xml")
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
            print(f"Deleted unmatched label: {lbl_path}")
    for file_stem in matched_files:
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            temp_path = os.path.join(images_dir, file_stem + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        if img_path is None:
            continue
        lbl_path = os.path.join(labels_dir, file_stem + ".xml")
        try:
            with Image.open(img_path) as img:
                img.load()
        except Exception as e:
            print(f"Corrupted image detected and deleted: {img_path} ({e})")
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='DroneVehicle')
    parser.add_argument('--input_dir', type=str, default='data/DroneVehicle')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    process_split(os.path.join(args.input_dir, "train_data"))
    process_split(os.path.join(args.input_dir, "test_data"))
    main_clean(os.path.join(args.input_dir, "train_data"))
    main_clean(os.path.join(args.input_dir, "test_data"))
    main_process(args)
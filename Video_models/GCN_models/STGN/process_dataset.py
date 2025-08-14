import os
import json
import numpy as np
from pathlib import Path
import argparse

def extract_coordinates_from_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        coordinates = []
        for key, value in data.items():
            if 'regions' in value:
                regions = value['regions']
                for region in regions:
                    if 'shape_attributes' in region:
                        shape_attrs = region['shape_attributes']
                        if 'x' in shape_attrs and 'y' in shape_attrs:
                            x = shape_attrs['x']
                            y = shape_attrs['y']
                            coordinates.append([x, y])
        if coordinates:
            return np.array(coordinates, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32).reshape(0, 2)
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return np.array([], dtype=np.float32).reshape(0, 2)

def process_dataset_split(data_dir, label_dir):
    data_path = Path(data_dir)
    label_path = Path(label_dir)
    if not data_path.exists():
        print(f"Data directory {data_dir} does not exist!")
        return
    label_path.mkdir(parents=True, exist_ok=True)
    for subfolder in data_path.iterdir():
        if subfolder.is_dir():
            print(f"Processing subfolder: {subfolder.name}")
            label_subfolder = label_path / subfolder.name
            label_subfolder.mkdir(parents=True, exist_ok=True)
            json_files = list(subfolder.glob("*.json"))
            for json_file in json_files:
                coordinates = extract_coordinates_from_json(json_file)
                npy_filename = json_file.stem + ".npy"
                npy_path = label_subfolder / npy_filename
                np.save(npy_path, coordinates)
                print(f"Created {npy_path} with {len(coordinates)} coordinate pairs")

def main(args):
    train_data_dir = os.path.join(args.input_dir, "train_data")
    train_label_dir = os.path.join(args.input_dir, "train_label")
    print("Processing training data...")
    process_dataset_split(train_data_dir, train_label_dir)
    test_data_dir = os.path.join(args.input_dir, "test_data")
    test_label_dir = os.path.join(args.input_dir, "test_label")
    print("\nProcessing test data...")
    process_dataset_split(test_data_dir, test_label_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='FDST')
    parser.add_argument('--input_dir', type=str, default='datasets/FDST')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)
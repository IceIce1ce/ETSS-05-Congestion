import os
import re
import argparse

def rename_folders(base_path):
    if not os.path.exists(base_path):
        print(f"Error: Path '{base_path}' does not exist!")
        return
    items = os.listdir(base_path)
    numeric_folders = []
    for item in items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and re.match(r'^\d+$', item):
            numeric_folders.append(item)
    if not numeric_folders:
        print("No numeric folders found to rename.")
        return
    numeric_folders.sort()
    renamed_count = 0
    for folder in numeric_folders:
        old_path = os.path.join(base_path, folder)
        new_name = f"Video{folder}"
        new_path = os.path.join(base_path, new_name)
        if os.path.exists(new_path):
            print(f"Warning: '{new_name}' already exists, skipping '{folder}'")
            continue
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: '{folder}' -> '{new_name}'")
            renamed_count += 1
        except OSError as e:
            print(f"Error renaming '{folder}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='UAVVIC')
    parser.add_argument('--input_dir', type=str, default='data/UAVVIC/frames')
    args = parser.parse_args()

    print('Rename dataset:', args.type_dataset)
    rename_folders(args.input_dir)

import random
from pathlib import Path
import argparse

def create_train_val_test_files(args):
    data_path = Path(args.input_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"
    if not train_path.exists():
        print(f"Error: Train directory '{train_path}' does not exist!")
        return
    if not val_path.exists():
        print(f"Error: Val directory '{val_path}' does not exist!")
        return
    train_subfolders = []
    if train_path.is_dir():
        train_subfolders = [item.name for item in train_path.iterdir() if item.is_dir()]
        train_subfolders.sort()
    val_subfolders = []
    if val_path.is_dir():
        val_subfolders = [item.name for item in val_path.iterdir() if item.is_dir()]
        val_subfolders.sort()
    random.seed(42)
    random.shuffle(val_subfolders)
    split_point = int(len(val_subfolders) * 0.8)
    val_list = val_subfolders[:split_point]
    test_list = val_subfolders[split_point:]
    train_file = data_path / "train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for subfolder in train_subfolders:
            f.write(f"Video{subfolder}\n")
    print(f"Created {train_file} with {len(train_subfolders)} entries")
    val_file = data_path / "val.txt"
    with open(val_file, 'w', encoding='utf-8') as f:
        for subfolder in val_list:
            f.write(f"Video{subfolder}\n")
    print(f"Created {val_file} with {len(val_list)} entries")
    test_file = data_path / "test.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        for subfolder in test_list:
            f.write(f"Video{subfolder}\n")
    print(f"Created {test_file} with {len(test_list)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='UAVVIC')
    parser.add_argument('--input_dir', type=str, default='data/UAVVIC')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    create_train_val_test_files(args)

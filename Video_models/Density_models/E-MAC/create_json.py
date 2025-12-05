import warnings
warnings.filterwarnings('ignore')
import os
import json
from pathlib import Path
import argparse

def create_json_files(dataset_root):
    dataset_root = Path(dataset_root)
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"Processing split: {split}")
        images_folder = dataset_root / split / 'images'
        ground_truth_folder = dataset_root / split / 'ground_truth'
        if not images_folder.exists():
            print(f"Warning: {images_folder} does not exist. Skipping {split}")
            continue
        if not ground_truth_folder.exists():
            print(f"Warning: {ground_truth_folder} does not exist. Skipping {split}")
            continue
        jpg_files = list(images_folder.glob('*.jpg'))
        if not jpg_files:
            print(f"Warning: No .jpg files found in {images_folder}")
            continue
        mat_files = list(ground_truth_folder.glob('GT_*.mat'))
        mat_basenames = {f.stem[3:] for f in mat_files}
        image_paths = []
        matched_count = 0
        unmatched_count = 0
        for jpg_file in jpg_files:
            jpg_basename = jpg_file.stem
            if jpg_basename in mat_basenames:
                relative_path = f"{split}/images/{jpg_file.name}"
                image_paths.append(relative_path)
                matched_count += 1
            else:
                unmatched_count += 1
                print(f"Warning: No ground truth GT_{jpg_basename}.mat found for {jpg_file.name}")
        image_paths.sort()
        sequences = {}
        for path in image_paths:
            filename = os.path.basename(path)
            if filename.startswith('img') and len(filename) >= 9:
                seq_id = filename[3:6]
                if seq_id not in sequences:
                    sequences[seq_id] = []
                sequences[seq_id].append(path)
        print(f"Found {len(image_paths)} matched image-groundtruth pairs")
        print(f"Found {unmatched_count} unmatched images")
        print(f"Detected {len(sequences)} sequences: {sorted(sequences.keys())}")
        for seq_id, seq_paths in sorted(sequences.items()):
            print(f"Sequence {seq_id}: {len(seq_paths)} images")
        json_file = dataset_root / f"{split}.json"
        with open(json_file, 'w') as f:
            json.dump(image_paths, f, indent=2)
        print(f"Created {json_file} with {len(image_paths)} entries")

def main(args):
    if not os.path.exists(args.input_dir):
        print(f"Error: Dataset root '{args.input_dir}' does not exist!")
        return
    create_json_files(args.input_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='DroneBird')
    parser.add_argument('--input_dir', type=str, default='datasets/DroneBird')
    args = parser.parse_args()

    print('Create json files for dataset:', args.type_dataset)
    main(args)
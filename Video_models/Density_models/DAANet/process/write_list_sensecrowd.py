import os
import random

if __name__ == '__main__':
    base_dir = "data/Sense/video_ori"
    output_dir = "data/Sense"
    subfolders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    train_folders = [d for d in subfolders if d.startswith("train_")]
    test_folders = [d for d in subfolders if d.startswith("test_")]
    random.shuffle(test_folders)
    split_idx = int(len(test_folders) * 0.7)
    val_folders = test_folders[:split_idx]
    real_test_folders = test_folders[split_idx:]
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for folder in sorted(train_folders):
            f.write(folder + "\n")
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        for folder in sorted(val_folders):
            f.write(folder + "\n")
    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        for folder in sorted(real_test_folders):
            f.write(folder + "\n")
    print("✅ Done! Files saved in", output_dir)
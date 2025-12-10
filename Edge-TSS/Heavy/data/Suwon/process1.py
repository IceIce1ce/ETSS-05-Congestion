import os
import shutil
import random

# Root folders in priority order
root_folders = ["hdmngoc", "joon", "phlong", "phson", "tdchi"]

# Destination folders
train_images_dir = os.path.join("train_data", "images")
train_txt_dir = os.path.join("train_data", "txt")
test_images_dir = os.path.join("test_data", "images")
test_txt_dir = os.path.join("test_data", "txt")

# Create destination directories
for folder in [train_images_dir, train_txt_dir, test_images_dir, test_txt_dir]:
    os.makedirs(folder, exist_ok=True)

# Keep track of video subfolders we have already processed
processed_videos = set()

# Collect all unique image-txt pairs
all_image_txt_pairs = []

for root_folder in root_folders:
    for folder_name, subfolders, _ in os.walk(root_folder):
        for video_subfolder in subfolders:
            if video_subfolder in processed_videos:
                continue  # skip duplicates
            video_path = os.path.join(folder_name, video_subfolder)
            images_dir = os.path.join(video_path, "images")
            labels_dir = os.path.join(video_path, "labels")
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue
            for image_file in os.listdir(images_dir):
                if image_file.endswith(".jpg") or image_file.endswith(".png"):
                    base_name = os.path.splitext(image_file)[0]
                    txt_file = base_name + ".txt"
                    txt_path = os.path.join(labels_dir, txt_file)
                    image_path = os.path.join(images_dir, image_file)
                    if os.path.exists(txt_path):
                        all_image_txt_pairs.append((image_path, txt_path))
            processed_videos.add(video_subfolder)

print(f"Total unique images collected: {len(all_image_txt_pairs)}")

# Shuffle and split
random.shuffle(all_image_txt_pairs)
split_idx = int(len(all_image_txt_pairs) * 0.8)
train_pairs = all_image_txt_pairs[:split_idx]
test_pairs = all_image_txt_pairs[split_idx:]

# Function to copy files
def copy_pairs(pairs, dest_images_dir, dest_txt_dir):
    for img_path, txt_path in pairs:
        shutil.copy(img_path, dest_images_dir)
        shutil.copy(txt_path, dest_txt_dir)

# Copy train and test data
copy_pairs(train_pairs, train_images_dir, train_txt_dir)
copy_pairs(test_pairs, test_images_dir, test_txt_dir)

print(f"Train images: {len(train_pairs)}, Test images: {len(test_pairs)}")

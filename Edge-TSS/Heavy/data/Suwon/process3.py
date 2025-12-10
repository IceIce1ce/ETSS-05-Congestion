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

# Keep track of processed video subfolders
processed_videos = set()

# Collect all unique video directories
all_videos = []

for root_folder in root_folders:
    for folder_name, subfolders, _ in os.walk(root_folder):
        for video_subfolder in subfolders:
            if video_subfolder in processed_videos:
                continue  # skip duplicates
            video_path = os.path.join(folder_name, video_subfolder)
            images_dir = os.path.join(video_path, "images")
            labels_dir = os.path.join(video_path, "labels")
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                all_videos.append((video_subfolder, images_dir, labels_dir))
                processed_videos.add(video_subfolder)

print(f"Total unique videos collected: {len(all_videos)}")

# Shuffle and split by videos
random.shuffle(all_videos)
split_idx = int(len(all_videos) * 0.8)
train_videos = all_videos[:split_idx]
test_videos = all_videos[split_idx:]

def copy_video_data(videos, dest_images_dir, dest_txt_dir):
    count = 0
    for video_name, images_dir, labels_dir in videos:
        for image_file in os.listdir(images_dir):
            if image_file.lower().endswith((".jpg", ".png")):
                base_name = os.path.splitext(image_file)[0]
                txt_file = base_name + ".txt"
                src_img_path = os.path.join(images_dir, image_file)
                src_txt_path = os.path.join(labels_dir, txt_file)
                if os.path.exists(src_txt_path):
                    shutil.copy(src_img_path, dest_images_dir)
                    shutil.copy(src_txt_path, dest_txt_dir)
                    count += 1
    return count

train_count = copy_video_data(train_videos, train_images_dir, train_txt_dir)
test_count = copy_video_data(test_videos, test_images_dir, test_txt_dir)

print(f"Train images copied: {train_count}, Test images copied: {test_count}")

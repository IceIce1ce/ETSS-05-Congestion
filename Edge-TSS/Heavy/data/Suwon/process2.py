import os
import cv2
import shutil

# Root folders to process
root_folders = ["train_data", "test_data"]

# Class IDs to ignore
ignore_class_ids = {0, 1, 2}

for root in root_folders:
    print(f"Processing {root}...")

    images_dir = os.path.join(root, "images")
    txt_dir = os.path.join(root, "txt")

    processed_images_dir = os.path.join(root, "processed_images")
    processed_txt_dir = os.path.join(root, "processed_txt")
    gt_show_dir = os.path.join(root, "gt_show")

    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(processed_txt_dir, exist_ok=True)
    os.makedirs(gt_show_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]

    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        image_file_jpg = os.path.splitext(txt_file)[0] + ".jpg"
        image_file_png = os.path.splitext(txt_file)[0] + ".png"

        image_path = None
        if os.path.exists(os.path.join(images_dir, image_file_jpg)):
            image_path = os.path.join(images_dir, image_file_jpg)
        elif os.path.exists(os.path.join(images_dir, image_file_png)):
            image_path = os.path.join(images_dir, image_file_png)
        else:
            print(f"Image not found for {txt_file}, skipping...")
            continue

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        filtered_points = []

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, w, h = parts
            class_id = int(class_id)
            if class_id in ignore_class_ids:
                continue
            x_center, y_center = float(x_center), float(y_center)
            x_pixel = int(round(x_center * img_width))
            y_pixel = int(round(y_center * img_height))
            filtered_points.append((x_pixel, y_pixel))

        # Only proceed if there is at least one valid point
        if filtered_points:
            # Write filtered TXT
            processed_txt_path = os.path.join(processed_txt_dir, txt_file)
            with open(processed_txt_path, "w") as f:
                for x, y in filtered_points:
                    f.write(f"{x}\t{y}\n")

            # Copy image to processed_images
            shutil.copy(image_path, processed_images_dir)

            # Draw points and save to gt_show
            img_copy = img.copy()
            for x, y in filtered_points:
                cv2.circle(img_copy, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
            gt_show_path = os.path.join(gt_show_dir, os.path.basename(image_path))
            cv2.imwrite(gt_show_path, img_copy)

    print(f"Finished processing {root}.")

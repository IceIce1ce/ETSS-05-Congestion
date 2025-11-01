import numpy as np
import os
from PIL import Image

def get_image_with_most_visible_pixels(folder_path):
    max_visible_pixels = -1
    best_image_filename = None
    best_mask = None
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".png")):
            continue
        filepath = os.path.join(folder_path, filename)
        try:
            with Image.open(filepath).convert("RGBA") as img:
                alpha = img.getchannel("A")
                alpha_data = alpha.load()
                width, height = img.size
                visible_count = sum(1 for x in range(width) for y in range(height) if alpha_data[x, y] > 127)
                if visible_count > max_visible_pixels:
                    max_visible_pixels = visible_count
                    best_image_filename = filename
                    best_mask = np.array(alpha)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
    return best_mask, best_image_filename

def find_nadir_image(folder):
    best_mask, best_name = get_image_with_most_visible_pixels(os.path.join(folder, "obj_seg"))
    image_dir = os.path.join(folder, "images_full_resolution") if os.path.isdir(os.path.join(folder, "images_full_resolution")) else os.path.join(folder, "images")
    nadir_image = Image.open(next(p for p in [os.path.join(image_dir, os.path.splitext(best_name)[0] + ext) for ext in [".png", ".jpg"]] if os.path.isfile(p)))
    best_mask = (best_mask).astype(np.uint8)
    mask_image = Image.fromarray(best_mask)
    mask_image = mask_image.resize(nadir_image.size, Image.BILINEAR)
    mask_image_np = np.array(mask_image) / 255.0
    mask = mask_image_np > 0.5
    coords = np.argwhere(mask)
    (y_min, x_min), (y_max, x_max) = coords.min(0), coords.max(0)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    pad_x = int(0.05 * bbox_width)
    pad_y = int(0.05 * bbox_height)
    x_min = max(0, x_min + pad_x)
    x_max = min(nadir_image.width, x_max - pad_x)
    y_min = max(0, y_min + pad_y)
    y_max = min(nadir_image.height, y_max - pad_y)
    print("Cropping", (x_min, y_min, x_max, y_max), "from", np.array(nadir_image).shape)
    cropped_nadir = nadir_image.crop((x_min, y_min, x_max, y_max))
    cropped_nadir.save(os.path.join(folder, "nadir.png"))
    print("Saved nadir to", os.path.join(folder, "nadir.png"))
    return np.array(cropped_nadir)
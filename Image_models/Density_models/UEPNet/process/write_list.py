import os

def get_image_files(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                if 'train_data' in directory:
                    relative_img_path = f"train_data/images/{filename}"
                    relative_txt_path = f"train_data/images/{filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')}"
                elif 'test_data' in directory:
                    relative_img_path = f"test_data/images/{filename}"
                    relative_txt_path = f"test_data/images/{filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')}"
                else:
                    relative_img_path = os.path.join(directory, filename)
                    relative_txt_path = relative_img_path.replace(os.path.splitext(filename)[1], '.txt')
                image_files.append((relative_img_path, relative_txt_path))
        image_files.sort()
    else:
        print(f"Warning: Directory {directory} does not exist")
    return image_files

def create_image_list_files():
    train_dir_a = "datasets/ShanghaiTech/part_A/train_data/images"
    test_dir_a = "datasets/ShanghaiTech/part_A/test_data/images"
    train_dir_b = "datasets/ShanghaiTech/part_B/train_data/images"
    test_dir_b = "datasets/ShanghaiTech/part_B/test_data/images"
    train_images_a = get_image_files(train_dir_a)
    test_images_a = get_image_files(test_dir_a)
    train_images_b = get_image_files(train_dir_b)
    test_images_b = get_image_files(test_dir_b)
    with open('train_sha.txt', 'w') as f:
        if train_images_a:
            for img_path, txt_path in train_images_a:
                f.write(f"{img_path} {txt_path}\n")
            print(f"Created train_sha.txt with {len(train_images_a)} image-txt mappings")
        else:
            print("No images found in part_A train directory")
    with open('test_sha.txt', 'w') as f:
        if test_images_a:
            for img_path, txt_path in test_images_a:
                f.write(f"{img_path} {txt_path}\n")
            print(f"Created test_sha.txt with {len(test_images_a)} image-txt mappings")
        else:
            print("No images found in part_A test directory")
    with open('train_shb.txt', 'w') as f:
        if train_images_b:
            for img_path, txt_path in train_images_b:
                f.write(f"{img_path} {txt_path}\n")
            print(f"Created train_shb.txt with {len(train_images_b)} image-txt mappings")
        else:
            print("No images found in part_B train directory")
    with open('test_shb.txt', 'w') as f:
        if test_images_b:
            for img_path, txt_path in test_images_b:
                f.write(f"{img_path} {txt_path}\n")
            print(f"Created test_shb.txt with {len(test_images_b)} image-txt mappings")
        else:
            print("No images found in part_B test directory")
    print("All files created successfully!")

if __name__ == "__main__":
    create_image_list_files()
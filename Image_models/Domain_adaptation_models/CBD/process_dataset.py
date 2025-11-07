import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.cm as cm

class CARPKDatasetCreator:
    def __init__(self, root_dir, args):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "Images"
        self.imagesets_dir = self.root_dir / "ImageSets"
        self.annotations_dir = self.root_dir / "VGG_annotation_truth"
        self.args = args

    def get_image_size(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        return img.shape[:2]

    def create_gaussian_density_map(self, points, img_height, img_width, sigma=15):
        density_map = np.zeros((img_height, img_width), dtype=np.float32)
        if len(points) == 0:
            return density_map
        h, w = img_height, img_width
        for point in points:
            x, y = int(point[0]), int(point[1])
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            gaussian = np.exp(-((np.arange(w) - x) ** 2 + (np.arange(h)[:, None] - y) ** 2) / (2 * sigma ** 2))
            gaussian /= (2 * np.pi * sigma ** 2)
            density_map += gaussian
        density_map = gaussian_filter(density_map, sigma=1)
        return density_map

    def parse_xml_annotations(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        points = []
        for obj in root.findall('object'):
            center_x = float(obj.find('point_2d/center_x').text)
            center_y = float(obj.find('point_2d/center_y').text)
            points.append([center_x, center_y])
        return np.array(points)

    def create_train_test_split(self):
        test_file = self.imagesets_dir / "test.txt"
        with open(test_file, 'r') as f:
            all_images = [line.strip() for line in f.readlines()]
        split_idx = int(0.8 * len(all_images))
        train_images = all_images[:split_idx]
        test_images = all_images[split_idx:]
        train_file = self.imagesets_dir / "train.txt"
        test_file_updated = self.imagesets_dir / "test_updated.txt"
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_images))
        with open(test_file_updated, 'w') as f:
            f.write('\n'.join(test_images))
        print(f"✅ Train: {len(train_images)} images")
        print(f"✅ Test:  {len(test_images)} images")
        return train_images, test_images

    def create_dataset_folder(self, split_name):
        dataset_dir = self.root_dir / split_name
        img_dir = dataset_dir / "img"
        den_dir = dataset_dir / "den"
        vis_dir = dataset_dir / "vis_gt"
        img_dir.mkdir(parents=True, exist_ok=True)
        den_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        return img_dir, den_dir, vis_dir

    def copy_images(self, image_names, img_dir):
        copied = 0
        for img_name in image_names:
            if self.args.type_dataset == 'CARPK':
                extension = 'png'
            elif self.args.type_dataset == 'PUCPR':
                extension = 'jpg'
            else:
                print('This dataset does not exist')
                raise NotImplementedError
            src_path = self.images_dir / f"{img_name}.{extension}"
            dst_path = img_dir / f"{img_name}.{extension}"
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1
        print(f"✅ Copied {copied}/{len(image_names)} images")

    def create_density_maps(self, image_names, den_dir):
        created = 0
        total_people = 0
        for img_name in image_names:
            if self.args.type_dataset == 'CARPK':
                extension = 'png'
            elif self.args.type_dataset == 'PUCPR':
                extension = 'jpg'
            else:
                print('This dataset does not exist')
                raise NotImplementedError
            img_path = self.images_dir / f"{img_name}.{extension}"
            if not img_path.exists():
                print(f"⚠️ Missing image: {img_name}.{extension}")
                continue
            h, w = self.get_image_size(img_path)
            xml_path = self.annotations_dir / f"{img_name}.xml"
            if not xml_path.exists():
                print(f"⚠️ Missing XML: {img_name}.xml")
                continue
            points = self.parse_xml_annotations(xml_path)
            total_people += len(points)
            density_map = self.create_gaussian_density_map(points, h, w, sigma=15)
            npy_path = den_dir / f"{img_name}.npy"
            np.save(npy_path, density_map)
            created += 1
            if created % 50 == 0:
                print(f"Progress: {created}/{len(image_names)}")
        print(f"✅ Created {created}/{len(image_names)} density maps")
        print(f"👥 Total people: {total_people}")

    def create_visual_overlay(self, img_path, density_map, output_path):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        density_norm = density_map / density_map.max()
        if density_map.max() == 0:
            density_norm = np.zeros_like(density_map)
        density_colored = cm.hot(density_norm)[:, :, :3]
        density_colored = (density_colored * 255).astype(np.uint8)
        alpha = 0.3
        overlay = (img_rgb.astype(float) * (1 - alpha) + density_colored.astype(float) * alpha).astype(np.uint8)
        people_count = int(density_map.sum())
        cv2.putText(overlay, f'People: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def create_visualizations(self, image_names, vis_dir):
        created = 0
        print("📸 Creating visualizations...")
        for img_name in image_names:
            if self.args.type_dataset == 'CARPK':
                extension = 'png'
            elif self.args.type_dataset == 'PUCPR':
                extension = 'jpg'
            else:
                print('This dataset does not exist')
                raise NotImplementedError
            img_path = self.images_dir / f"{img_name}.{extension}"
            den_path = vis_dir.parent / "den" / f"{img_name}.npy"
            if not img_path.exists() or not den_path.exists():
                continue
            density_map = np.load(den_path)
            output_path = vis_dir / f"{img_name}.{extension}"
            self.create_visual_overlay(img_path, density_map, output_path)
            created += 1
            if created % 50 == 0:
                print(f"Progress: {created}/{len(image_names)}")
        print(f"✅ Created {created}/{len(image_names)} visualizations")

    def process_split(self, split_name, image_names):
        print(f"\n{'=' * 50}")
        print(f"🚀 Processing {split_name.upper()}...")
        print(f"{'=' * 50}")
        img_dir, den_dir, vis_dir = self.create_dataset_folder(split_name)
        self.copy_images(image_names, img_dir)
        self.create_density_maps(image_names, den_dir)
        self.create_visualizations(image_names, vis_dir)

    def run(self):
        train_images, test_images = self.create_train_test_split()
        self.process_split("train", train_images)
        self.process_split("test", test_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='CARPK')
    parser.add_argument('--input_dir', type=str, default='datasets/CARPK')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    creator = CARPKDatasetCreator(args.input_dir, args)
    creator.run()
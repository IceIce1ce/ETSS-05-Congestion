"""
Official PyTorch Dataset implementation for the TPC-268 Dataset.

This script provides the `TPC268Dataset` class, which handles:
1. Loading image paths from dataset split files (e.g., train.txt, test.txt).
2. Parsing the global JSON annotation file to extract instances (points) and exemplars (4-point polygons).
3. Automatically converting 4-point polygons into standard horizontal bounding boxes [x_min, y_min, x_max, y_max]
   for maximum compatibility with mainstream object detection and counting frameworks.
4. Returning standardized PyTorch Tensors.

Usage Example:
    from tools.dataset import TPC268Dataset

    dataset = TPC268Dataset(
        data_dir='./data',
        split_txt='./splits/train.txt',
        anno_json='./annotations/tpc268_annotations.json'
    )
    img, target = dataset[0]
    print("Points shape:", target['points'].shape) # torch.Size([N, 2])
    print("Boxes shape:", target['boxes'].shape)   # torch.Size([M, 4])
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class TPC268Dataset(Dataset):
    def __init__(self, data_dir, split_txt, anno_json, transform=None):
        """
        Args:
            data_dir (str): Root directory where the unzipped images are stored (e.g., './data').
            split_txt (str): Path to the txt file containing relative image paths (e.g., './splits/train.txt').
            anno_json (str): Path to the global annotation JSON file.
            transform (callable, optional): Optional transform to be applied on a sample (image and target).
        """
        self.data_dir = data_dir
        self.transform = transform

        # 1. Load relative image paths from the split text file
        if not os.path.exists(split_txt):
            raise FileNotFoundError(f"Split file not found: {split_txt}")

        with open(split_txt, "r") as f:
            # e.g., "Abelmoschus_esculentus/fruit/Abelmoschus_esculentus_fruit_1.jpg"
            self.img_paths = [line.strip() for line in f.readlines() if line.strip()]

        # 2. Load the global annotation JSON
        if not os.path.exists(anno_json):
            raise FileNotFoundError(f"Annotation JSON not found: {anno_json}")

        with open(anno_json, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        """Returns the total number of images in this dataset split."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Fetches the image and its corresponding annotations at the given index.

        Returns:
            image (PIL.Image or Tensor): The input image (RGB).
            target (dict): A dictionary containing:
                - 'points' (Tensor): Instance point coordinates of shape [N, 2].
                - 'boxes' (Tensor): Standard bounding boxes [x_min, y_min, x_max, y_max] of shape [M, 4].
                - 'polygons' (Tensor): Original 4-point polygons of shape [M, 4, 2].
                - 'image_id' (str): The pure filename of the image.
        """
        img_rel_path = self.img_paths[idx]
        img_path = os.path.join(self.data_dir, img_rel_path)

        # 1. Load image (convert to RGB to ensure 3 channels consistently)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image at {img_path}. Error: {e}")

        # 2. Retrieve annotations using the pure image filename as the key
        img_filename = os.path.basename(img_rel_path)
        anno = self.annotations.get(img_filename, {})

        # 3. Process instance points (Instances)
        points = anno.get("points", [])
        if len(points) > 0:
            points_tensor = torch.as_tensor(points, dtype=torch.float32)
        else:
            # Handle empty points safely to prevent shape mismatch errors in PyTorch
            points_tensor = torch.empty((0, 2), dtype=torch.float32)

        # 4. Process exemplar bounding boxes (Exemplars)
        polygons = anno.get("box_examples_coordinates", [])
        boxes = []
        for poly in polygons:
            # Parse the 4-point polygon format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            # Convert to standard horizontal bounding box: [x_min, y_min, x_max, y_max]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])

        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            polygons_tensor = torch.as_tensor(polygons, dtype=torch.float32)
        else:
            # Handle empty boxes safely
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            polygons_tensor = torch.empty((0, 4, 2), dtype=torch.float32)

        # 5. Assemble the standardized target dictionary
        target = {
            "points": points_tensor,
            "boxes": boxes_tensor,
            "polygons": polygons_tensor,
            "image_id": img_filename,
        }

        # 6. Apply transforms (Data Augmentation) if specified
        if self.transform is not None:
            # Note: A custom transform function must handle both the PIL image
            # and the target dictionary (updating point/box coordinates accordingly).
            image, target = self.transform(image, target)

        return image, target


if __name__ == "__main__":
    # Simple sanity check message
    print("TPC268Dataset module loaded successfully. Ready for integration.")

"""
=============================================================================
TPC-268 Dataset Visualization Tool
=============================================================================

This script visualizes the point and bounding box annotations for a given 
image in the TPC-268 dataset. It reads the annotation JSON, overlays the 
instances (points) and exemplars (boxes) onto the image, and saves the 
result to an output directory.

Usage Example:
    python visualize_dataset.py \
        --img_path data_final/Abelmoschus_esculentus/fruit/Abelmoschus_esculentus_fruit_1.jpg \
        --anno_json annotations/tpc268_annotations.json \
        --output_dir vis_results

Arguments:
    --img_path    : Path to the specific image you want to visualize.
    --anno_json   : Path to the TPC-268 annotation JSON file.
    --output_dir  : Directory where the visualized image will be saved.
    --point_color : Color of the point annotations (B, G, R). Default is Yellow (0, 255, 255).
    --box_color   : Color of the bounding boxes (B, G, R). Default is Red (0, 0, 255).
=============================================================================
"""

import os
import json
import argparse
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize annotations for TPC-268 Dataset")
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--anno_json", type=str, default="annotations/tpc268_annotations.json",
                        help="Path to the dataset annotation JSON.")
    parser.add_argument("--output_dir", type=str, default="vis_results",
                        help="Directory to save the visualization results.")
    
    # Optional arguments for customizable drawing aesthetics (OpenCV uses BGR format)
    parser.add_argument("--point_color", type=int, nargs=3, default=[0, 255, 255],
                        help="Point color in BGR format (default: Yellow).")
    parser.add_argument("--box_color", type=int, nargs=3, default=[0, 0, 255],
                        help="Bounding box color in BGR format (default: Red).")
    parser.add_argument("--radius", type=int, default=4,
                        help="Radius of the annotation points.")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Line thickness for the bounding boxes.")
    
    return parser.parse_args()


def visualize(args):
    # 1. Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Check if the input image exists
    if not os.path.exists(args.img_path):
        print(f"Error: Image not found at {args.img_path}")
        return

    # 3. Load the annotation JSON
    print(f"Loading annotations from {args.anno_json}...")
    try:
        with open(args.anno_json, "r") as f:
            anno_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {args.anno_json}")
        return

    # Extract the filename as the key to search in the JSON
    img_filename = os.path.basename(args.img_path)
    if img_filename not in anno_data:
        print(f"Error: No annotations found for '{img_filename}' in the JSON file.")
        return

    img_annos = anno_data[img_filename]
    points = img_annos.get("points", [])
    boxes = img_annos.get("box_examples_coordinates", [])

    print(f"Found {len(points)} points and {len(boxes)} exemplar boxes for {img_filename}.")

    # 4. Read the image using OpenCV
    img = cv2.imread(args.img_path)
    if img is None:
        print(f"Error: Failed to read the image at {args.img_path}. It might be corrupted.")
        return

    # Convert color arguments to tuples
    point_color = tuple(args.point_color)
    box_color = tuple(args.box_color)

    # 5. Draw bounding boxes (Exemplars)
    # The JSON provides 4 corners for each box: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    for box in boxes:
        # Convert the list of coordinates into a NumPy array of shape (4, 1, 2) for OpenCV
        pts_array = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_array], isClosed=True, color=box_color, thickness=args.thickness)

    # 6. Draw point annotations (Instances)
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), radius=args.radius, color=point_color, thickness=-1)

    # 7. Save the visualized image
    output_filename = f"vis_{img_filename}"
    output_path = os.path.join(args.output_dir, output_filename)
    
    cv2.imwrite(output_path, img)
    print(f"Visualization saved successfully to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    visualize(args)
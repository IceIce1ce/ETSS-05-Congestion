import os
import re
from pathlib import Path
import cv2
from natsort import natsorted

def filter_and_sort_images(folder_path):
    """
    Filter images matching pattern image-x-xxxxxx.jpg and sort them.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        List of sorted image file paths
    """
    # Pattern to match: image-{digit(s)}-{6 digits}.jpg
    pattern = re.compile(r'^image-\d+-\d{6}\.jpg$', re.IGNORECASE)
    
    # Get all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter files matching the pattern
    matched_files = [f for f in all_files if pattern.match(f)]
    
    # Sort files naturally (image-1-000001.jpg, image-1-000002.jpg, etc.)
    sorted_files = natsorted(matched_files)
    
    # Create full paths
    image_paths = [os.path.join(folder_path, f) for f in sorted_files]
    
    print(f"Found {len(image_paths)} matching images out of {len(all_files)} total files")
    print(f"First few images: {sorted_files[:5]}")
    print(f"Last few images: {sorted_files[-5:]}")
    
    return image_paths

def create_video_from_images(image_paths, output_path, fps=30):
    """
    Create an MP4 video from a list of images.
    
    Args:
        image_paths: List of image file paths
        output_path: Path for the output video file
        fps: Frames per second for the video (default: 30)
    """
    if not image_paths:
        print("No images to process!")
        return
    
    # Read the first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"Error: Could not read first image {image_paths[0]}")
        return
    
    height, width, channels = first_img.shape
    print(f"Video dimensions: {width}x{height}")
    print(f"Creating video with {len(image_paths)} frames at {fps} FPS")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1' for H.264
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image as a frame
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping...")
            continue
        
        # Resize if dimensions don't match
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
        
        # Show progress
        if (i + 1) % 50 == 0 or i == len(image_paths) - 1:
            print(f"Progress: {i + 1}/{len(image_paths)} frames processed")
    
    # Release the video writer
    video_writer.release()
    print(f"\nVideo successfully created: {output_path}")

def main():
    # Configuration
    IMAGES_FOLDER = "images"  # Change this to your folder path
    OUTPUT_VIDEO = "output_video.mp4"
    FPS = 15  # Adjust frames per second as needed
    
    # Check if folder exists
    if not os.path.exists(IMAGES_FOLDER):
        print(f"Error: Folder '{IMAGES_FOLDER}' not found!")
        return
    
    # Filter and sort images
    image_paths = filter_and_sort_images(IMAGES_FOLDER)
    
    if not image_paths:
        print("No matching images found!")
        return
    
    # Create video
    create_video_from_images(image_paths, OUTPUT_VIDEO, fps=FPS)

if __name__ == "__main__":
    main()

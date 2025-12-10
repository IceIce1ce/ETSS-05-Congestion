import os.path
from pathlib import Path
from PIL import Image
import cv2
import argparse
from multiprocessing import Pool, cpu_count
import subprocess

def convert_mp4_to_gif_ffmpeg(mp4_path, output_path, fps=10, scale=None):
    """
    High-quality conversion using ffmpeg with proper palette generation
    
    Args:
        mp4_path: Path to input MP4
        output_path: Path to output GIF
        fps: Frame rate for output GIF (default 10)
        scale: Scale filter, e.g., "320:-1" for width 320, or None for original size
    """
    # Two-pass encoding for better quality
    # Pass 1: Generate palette
    palette_path = str(output_path).replace('.gif', '_palette.png')
    
    # Build filter for palette generation
    filters = []
    if fps:
        filters.append(f'fps={fps}')
    if scale:
        filters.append(f'scale={scale}:flags=lanczos')
    
    # Add palette generation (removes grid artifacts)
    filters.append('palettegen=stats_mode=diff')
    
    palette_cmd = [
        'ffmpeg', '-i', str(mp4_path), '-y',
        '-vf', ','.join(filters),
        palette_path
    ]
    
    try:
        subprocess.run(palette_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return False, f"Palette generation failed: {e.stderr}"
    
    # Pass 2: Use palette to create high-quality GIF
    # Build the complex filter properly
    filter_complex = []
    
    # First part: process input video
    video_filters = []
    if fps:
        video_filters.append(f'fps={fps}')
    if scale:
        video_filters.append(f'scale={scale}:flags=lanczos')
    
    if video_filters:
        filter_complex.append(','.join(video_filters) + '[x]')
    else:
        filter_complex.append('[0:v][x]')  # Just pass through
    
    # Second part: apply palette with dithering
    filter_complex.append('[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle')
    
    gif_cmd = [
        'ffmpeg', '-i', str(mp4_path), '-i', palette_path, '-y',
        '-lavfi', ';'.join(filter_complex),
        str(output_path)
    ]
    
    try:
        subprocess.run(gif_cmd, capture_output=True, text=True, check=True)
        # Clean up palette file
        if os.path.exists(palette_path):
            os.remove(palette_path)
        return True, None
    except subprocess.CalledProcessError as e:
        # Clean up palette file even on error
        if os.path.exists(palette_path):
            os.remove(palette_path)
        return False, f"GIF creation failed: {e.stderr}"

def convert_mp4_to_gif_opencv(mp4_path, output_path, max_frames=None, skip_frames=1, scale_factor=1.0):
    """
    Optimized OpenCV conversion with frame skipping and scaling options
    
    Args:
        mp4_path: Path to input MP4
        output_path: Path to output GIF
        max_frames: Maximum frames to use (None = all)
        skip_frames: Skip every N frames (1 = use all, 2 = use every other frame)
        scale_factor: Scale images (1.0 = original, 0.5 = half size)
    """
    cap = cv2.VideoCapture(str(mp4_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is None:
        frames_to_use = total_frames
    else:
        frames_to_use = min(total_frames, max_frames)
    
    print(f"Process: {mp4_path.name} ({total_frames} frames, skip={skip_frames}, scale={scale_factor})")
    
    frames = []
    frame_count = 0
    processed_count = 0
    
    while frame_count < frames_to_use:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for speed
        if frame_count % skip_frames == 0:
            # Scale down if needed
            if scale_factor != 1.0:
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            processed_count += 1
        
        frame_count += 1
    
    cap.release()
    
    if frames:
        # Optimize GIF saving
        frames[0].save(
            output_path, 
            save_all=True, 
            append_images=frames[1:], 
            duration=100, 
            loop=0,
            optimize=False  # Faster but larger files
        )
        print(f"✓ Saved: {output_path.name} ({processed_count} frames)")
        return True
    else:
        print(f"✗ Warning: No frames extracted from {mp4_path.name}")
        return False

def process_single_video_ffmpeg(args_tuple):
    """Wrapper for multiprocessing with ffmpeg"""
    mp4_file, output_dir, fps, scale = args_tuple
    gif_filename = mp4_file.stem + ".gif"
    output_path = output_dir / gif_filename
    
    print(f"Processing: {mp4_file.name}")
    success, error = convert_mp4_to_gif_ffmpeg(mp4_file, output_path, fps=fps, scale=scale)
    
    if success:
        print(f"✓ Completed: {gif_filename}")
    else:
        print(f"✗ Failed: {gif_filename} - {error}")
    
    return success

def process_single_video_opencv(args_tuple):
    """Wrapper for multiprocessing with opencv"""
    mp4_file, output_dir, max_frames, skip_frames, scale_factor = args_tuple
    gif_filename = mp4_file.stem + ".gif"
    output_path = output_dir / gif_filename
    
    success = convert_mp4_to_gif_opencv(mp4_file, output_path, max_frames, skip_frames, scale_factor)
    return success

def main(args):
    source_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get all MP4 files directly from the input directory
    mp4_files = list(source_dir.glob("*.mp4"))
    
    if not mp4_files:
        print(f"No MP4 files found in {source_dir}")
        return
    
    print(f"Found {len(mp4_files)} MP4 files in {source_dir}")
    print(f"Method: {args.method}")
    print(f"Workers: {args.workers}")
    print("="*60)
    
    if args.method == 'ffmpeg':
        # Use ffmpeg (FASTEST method)
        print("Using ffmpeg (fastest method)")
        
        if args.parallel:
            # Parallel processing
            process_args = [(f, output_dir, args.fps, args.scale) for f in mp4_files]
            with Pool(processes=args.workers) as pool:
                results = pool.map(process_single_video_ffmpeg, process_args)
            success_count = sum(results)
        else:
            # Sequential processing
            success_count = 0
            for mp4_file in mp4_files:
                gif_filename = mp4_file.stem + ".gif"
                output_path = output_dir / gif_filename
                success, error = convert_mp4_to_gif_ffmpeg(mp4_file, output_path, fps=args.fps, scale=args.scale)
                if success:
                    print(f"✓ Completed: {gif_filename}")
                    success_count += 1
                else:
                    print(f"✗ Failed: {gif_filename}")
    
    else:
        # Use OpenCV (slower but more control)
        print(f"Using OpenCV (skip_frames={args.skip_frames}, scale={args.scale_factor})")
        
        if args.parallel:
            # Parallel processing
            process_args = [(f, output_dir, args.max_frames, args.skip_frames, args.scale_factor) for f in mp4_files]
            with Pool(processes=args.workers) as pool:
                results = pool.map(process_single_video_opencv, process_args)
            success_count = sum(results)
        else:
            # Sequential processing
            success_count = 0
            for mp4_file in mp4_files:
                gif_filename = mp4_file.stem + ".gif"
                output_path = output_dir / gif_filename
                if convert_mp4_to_gif_opencv(mp4_file, output_path, args.max_frames, args.skip_frames, args.scale_factor):
                    success_count += 1
    
    print("="*60)
    print(f"Completed: {success_count}/{len(mp4_files)} videos converted successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert MP4 to GIF (Fast)')
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--input_dir', type=str, default='vis')
    parser.add_argument('--output_dir', type=str, default='saved_sense_gif')
    
    # Method selection
    parser.add_argument('--method', type=str, default='ffmpeg', choices=['ffmpeg', 'opencv'],
                        help='Conversion method: ffmpeg (fastest) or opencv (more control)')
    
    # FFmpeg options (for method='ffmpeg')
    parser.add_argument('--fps', type=int, default=10, help='Frame rate for output GIF (ffmpeg method)')
    parser.add_argument('--scale', type=str, default=None, 
                        help='Scale filter for ffmpeg, e.g., "320:-1" for width 320, "640:480" for specific size')
    
    # OpenCV options (for method='opencv')
    parser.add_argument('--max_frames', type=int, default=None, 
                        help='Maximum frames to use (opencv method, None = use all frames)')
    parser.add_argument('--skip_frames', type=int, default=1, 
                        help='Skip every N frames (opencv method, 1=use all, 2=every other frame, etc.)')
    parser.add_argument('--scale_factor', type=float, default=1.0,
                        help='Scale factor for images (opencv method, 0.5=half size, 1.0=original)')
    
    # Performance options
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Use parallel processing (multiple videos at once)')
    parser.add_argument('--workers', type=int, default=cpu_count()//2,
                        help='Number of parallel workers (default: half of CPU cores)')
    
    args = parser.parse_args()
    
    print('Process dataset:', args.type_dataset)
    if args.max_frames is None and args.method == 'opencv':
        print('Mode: Using ALL frames from videos')
    
    main(args)

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from yolo_poser.utils import FFmpegWriter, get_device, load_yolo_model


def calculate_crop_params(video_path, bbox, padding=0.1, keep_proportions=True):
    """Calculate crop parameters with separate horizontal/vertical padding"""
    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    min_x, min_y, max_x, max_y = bbox
    
    # Calculate dimensions
    width = max_x - min_x
    height = max_y - min_y
    
    # Apply different padding rules
    # More padding horizontally (since movement is mostly side-to-side)
    # Less padding vertically (since vertical size is already large)
    horizontal_padding = width * padding
    vertical_padding = height * (padding * 0.5)  # Half the horizontal padding
    
    # Apply padding
    min_x = max(0, min_x - horizontal_padding)
    max_x = min(orig_width, max_x + horizontal_padding)
    min_y = max(0, min_y - vertical_padding)
    max_y = min(orig_height, max_y + vertical_padding)
    
    target_width = max_x - min_x
    target_height = max_y - min_y
    
    if keep_proportions:
        # Maintain original aspect ratio
        orig_aspect = orig_width / orig_height
        crop_aspect = target_width / target_height
        
        if crop_aspect > orig_aspect:
            target_height = target_width / orig_aspect
        else:
            target_width = target_height * orig_aspect
    
    # Center the crop box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Calculate final crop parameters
    crop_x = max(0, int(center_x - target_width/2))
    crop_y = max(0, int(center_y - target_height/2))
    crop_width = int(min(target_width, orig_width - crop_x))
    crop_height = int(min(target_height, orig_height - crop_y))
    
    return crop_x, crop_y, crop_width, crop_height


def extract_frames(video_path, num_frames):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    
    cap.release()
    return frames


def get_combined_bbox(video_path, num_samples=30, outlier_threshold=1.5, debug=False):
    """Get bounding box that encompasses all detected people, excluding outliers"""
    if debug:
        print("\nStarting person detection and bbox calculation...")
        print(f"Processing {num_samples} frames from video...")
    
    model = load_yolo_model()
    frames = extract_frames(video_path, num_samples)
    
    # Store all coordinates separately for statistical analysis
    xs_min, ys_min, xs_max, ys_max = [], [], [], []
    valid_frames = []
    debug_frames = []
    
    if debug:
        print("\nAnalyzing frames for person detection...")
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)
    
    # First pass: collect all coordinates
    for i, (frame_idx, frame) in enumerate(frames):
        if debug and i % 5 == 0:  # Status update every 5 frames
            print(f"Processing frame {i+1}/{len(frames)}...")
            
        results = model.predict(frame, classes=0)
        
        if len(results[0].boxes) == 1:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            xs_min.append(box[0])
            ys_min.append(box[1])
            xs_max.append(box[2])
            ys_max.append(box[3])
            valid_frames.append((frame_idx, frame, box))
    
    if debug:
        print(f"\nFound {len(valid_frames)} frames with single person detection")
        print("\nRemoving outliers...")
    
    # Calculate IQR and bounds for each coordinate
    def remove_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - outlier_threshold * iqr
        upper_bound = q3 + outlier_threshold * iqr
        return lower_bound, upper_bound
    
    # Get bounds for each coordinate
    x_min_bounds = remove_outliers(xs_min)
    y_min_bounds = remove_outliers(ys_min)
    x_max_bounds = remove_outliers(xs_max)
    y_max_bounds = remove_outliers(ys_max)
    
    # Filter frames based on bounds
    filtered_frames = []
    for frame_idx, frame, box in valid_frames:
        if (x_min_bounds[0] <= box[0] <= x_min_bounds[1] and
            y_min_bounds[0] <= box[1] <= y_min_bounds[1] and
            x_max_bounds[0] <= box[2] <= x_max_bounds[1] and
            y_max_bounds[0] <= box[3] <= y_max_bounds[1]):
            filtered_frames.append((frame_idx, frame, box))
    
    # Calculate final bounding box from filtered frames
    if filtered_frames:
        if debug:
            print(f"Calculating final bbox from {len(filtered_frames)} filtered frames...")
        min_x = np.median([box[0] for _, _, box in filtered_frames])
        min_y = np.median([box[1] for _, _, box in filtered_frames])
        max_x = np.median([box[2] for _, _, box in filtered_frames])
        max_y = np.median([box[3] for _, _, box in filtered_frames])
    else:
        raise ValueError("No valid frames after filtering outliers")
    
    if debug:
        print("\nGenerating debug visualizations...")
        for frame_idx, frame in frames:
            debug_frame = frame.copy()
            results = model.predict(frame, classes=0)
            
            # Add frame info
            cv2.putText(debug_frame, 
                       f"Frame {frame_idx}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (255, 255, 255), 
                       2)
            
            # Draw all detections
            if len(results[0].boxes) == 1:
                box = results[0].boxes.xyxy[0].cpu().numpy()
                
                # Determine if this frame was filtered as outlier
                is_outlier = not any(f[0] == frame_idx for f in filtered_frames)
                color = (0, 0, 255) if is_outlier else (0, 255, 0)
                
                cv2.rectangle(debug_frame, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            color, 
                            2)
                
                # Add outlier status
                status = "OUTLIER" if is_outlier else "VALID"
                cv2.putText(debug_frame, 
                           status, 
                           (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           color, 
                           2)
            
            debug_path = debug_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)
            debug_frames.append(debug_frame)
        
        # Create summary visualization
        summary_frame = frames[len(frames)//2][1].copy()
        cv2.rectangle(summary_frame,
                     (int(min_x), int(min_y)),
                     (int(max_x), int(max_y)),
                     (0, 255, 255),
                     2)
        
        cv2.putText(summary_frame,
                   f"Final bbox (after outlier removal): {int(max_x-min_x)}x{int(max_y-min_y)}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   (0, 255, 255),
                   2)
        
        cv2.imwrite(str(debug_dir / "summary.jpg"), summary_frame)
        
        print(f"\nFinal bbox dimensions: {int(max_x-min_x)}x{int(max_y-min_y)}")
        print(f"Debug frames saved to: {debug_dir}/")
        print(f"Valid frames after outlier removal: {len(filtered_frames)}/{len(valid_frames)}")
    
    return (min_x, min_y, max_x, max_y), filtered_frames

def visualize_crop(frame, crop_params, output_path=None):
    """Visualize proposed crop on a frame"""
    x, y, w, h = crop_params
    vis_frame = frame.copy()
    
    # Draw crop rectangle
    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add text with dimensions
    text = f'Crop: {w}x{h}'
    cv2.putText(vis_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_frame)
    else:
        # Show interactive window
        cv2.imshow('Proposed Crop', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def crop_video(input_path, output_path, crop_params, debug=False):
    """Apply the crop using ffmpeg"""
    x, y, w, h = crop_params
    
    # Ensure width and height are even numbers
    w = w - (w % 2)
    h = h - (h % 2)
    
    # Construct FFmpeg command to crop video while preserving audio
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', f'crop={w}:{h}:{x}:{y}',
        '-c:a', 'copy',  # Copy audio stream without re-encoding
        '-pix_fmt', 'yuv420p',  # Ensure QuickTime compatibility
        str(output_path)
    ]
    
    try:
        print(f"\nCropping video with dimensions {w}x{h}...")
        if debug:
            print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            print("\nFFmpeg output:")
            print(result.stderr)  # FFmpeg writes its progress to stderr
        else:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
        print("\nCropping completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during video cropping: {e.stderr.decode()}")
        raise

def main(input_video=None, padding=0.3, keep_proportions=True, preview=True, debug=False, output=None):
    """Main function with configurable options"""
    if input_video is None:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Auto-crop video based on person detection')
        parser.add_argument('input_video', help='Path to input video')
        parser.add_argument('--padding', type=float, default=0.5, help='Padding around detected area (default: 0.5)')
        parser.add_argument('--no-keep-proportions', action='store_false', dest='keep_proportions',
                          help='Do not maintain original video proportions')
        parser.add_argument('--no-preview', action='store_false', dest='preview',
                          help='Skip preview and confirmation')
        parser.add_argument('--debug', action='store_true', help='Save debug frames showing detections')
        parser.add_argument('--output', help='Output video path (default: input_cropped.mp4)')
        
        args = parser.parse_args()
        
        return main(
            input_video=args.input_video,
            padding=args.padding,
            keep_proportions=args.keep_proportions,
            preview=args.preview,
            debug=args.debug,
            output=args.output
        )
    
    # Get bounding box and sample frames
    bbox, valid_frames = get_combined_bbox(input_video, debug=debug)
    
    if not valid_frames:
        print("No valid frames found with single person detection")
        return
    
    # Calculate crop parameters
    crop_params = calculate_crop_params(input_video, bbox, padding, keep_proportions)
    
    if preview:
        # Use middle frame for preview
        middle_frame_data = valid_frames[len(valid_frames)//2]
        frame_idx, frame, detection = middle_frame_data
        
        # Show original detection
        cv2.rectangle(frame, 
                     (int(detection[0]), int(detection[1])), 
                     (int(detection[2]), int(detection[3])), 
                     (255, 0, 0), 2)
        
        # Create preview directory if it doesn't exist
        preview_dir = Path("crop_previews")
        preview_dir.mkdir(exist_ok=True)
        
        # Save preview
        preview_path = preview_dir / f"crop_preview_{Path(input_video).stem}.jpg"
        visualize_crop(frame, crop_params, str(preview_path))
        print(f"\nPreview saved to: {preview_path}")
        
        # Ask for confirmation
        response = input("Proceed with crop? (y/n): ")
        if response.lower() != 'y':
            print("Crop cancelled")
            return
    
    # Apply crop
    if output is None:
        output = f"{Path(input_video).stem}_cropped{Path(input_video).suffix}"
    crop_video(input_video, output, crop_params, debug)
    print(f"Cropped video saved as: {output}")

if __name__ == "__main__":
    main()
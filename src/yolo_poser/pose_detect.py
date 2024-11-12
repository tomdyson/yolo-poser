#!/usr/bin/env python3

"""
Pose Detection Script
====================

This script processes video files using YOLO pose detection to track human body keypoints.
It outputs a visualization video with pose overlay.

Dependencies
-----------
- Python 3.8+
- torch: PyTorch for neural network processing
- ultralytics: YOLO implementation
- opencv-python (cv2): Video processing and visualization
- numpy: Numerical operations
- subprocess: For managing FFmpeg subprocess
- sync_audio: For audio syncing

Installation
-----------
pip install torch ultralytics opencv-python numpy sync_audio

Usage
-----
Basic video processing:
    python pose-detect.py input_video.mp4

Arguments:
    input             Input video file path
    --model          Path to YOLO model (default: yolo11n-pose.pt)
    --output         Output video path (default: input_pose_detected.mp4)
    --output-format  Output video format: 'mjpeg' or 'h264' or 'webm' (default: h264)
    --debug          Output timing information

Example:
    python pose-detect.py input.mp4 --model custom_model.pt --output result.mp4 --output-format h264

Tracked Keypoints
---------------
- Eyes: left_eye, right_eye
- Upper body: left/right_shoulder, left/right_elbow, left/right_wrist
- Lower body: left/right_hip, left/right_knee, left/right_ankle

Visualization Features
-------------------
- Color coding: Green (eyes), Pink (arms), Orange (legs)
- Confidence threshold: Points below threshold are not displayed
- Smoothing: Exponential smoothing to reduce jitter
- Transparent overlays for better visibility

Output Formats
------------
- H.264 (.mp4): Default format, efficient compression using FFmpeg
- MJPEG (.avi): Alternative format for compatibility
- WebM (.webm): Alternative format using VP9 codec
"""

import argparse
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

import cv2
import numpy as np

from .sync_audio import sync_audio
from .utils import FFmpegWriter, get_device, load_yolo_model

# Detection and smoothing thresholds
CONFIDENCE_THRESHOLD = 0.3
SMOOTHING_CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.4

# Visualization settings
LINE_OPACITY = 1
POINT_OPACITY = 1
LINE_WIDTH = 6  # Line thickness in pixels
POINT_RADIUS = 4  # Point radius in pixels



class TimingStats:
    def __init__(self):
        self.timings = defaultdict(list)
        
    def add_timing(self, operation, duration):
        self.timings[operation].append(duration)
    
    def get_summary(self):
        summary = []
        for operation, times in self.timings.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            total_time = sum(times)
            fps = len(times) / total_time if total_time > 0 else 0
            
            summary.append(f"{operation}:")
            summary.append(f"  Average: {avg_time*1000:.2f}ms")
            summary.append(f"  Min: {min_time*1000:.2f}ms")
            summary.append(f"  Max: {max_time*1000:.2f}ms")
            summary.append(f"  Total: {total_time:.2f}s")
            summary.append(f"  Frames: {len(times)}")
            summary.append(f"  FPS: {fps:.1f}")
            summary.append("")
        return "\n".join(summary)

@contextmanager
def timer(stats: TimingStats, operation: str):
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    stats.add_timing(operation, duration)

# Output format configurations
OUTPUT_FORMATS = {
    'mjpeg': {
        'fourcc': 'MJPG',
        'ext': '.avi'
    },
    'h264': {
        'fourcc': 'avc1',  # or 'H264' on some systems
        'ext': '.mp4'
    },
    'webm': {
        'fourcc': 'VP90',  # Not used for FFmpeg but kept for consistency
        'ext': '.webm'
    }
}

# Keypoint configurations
KEYPOINT_NAMES = [
    "left_eye", "right_eye",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Define which keypoints to keep (eyes and body, skip nose and ears)
KEPT_KEYPOINTS = [1, 2] + list(range(5, 17))  # Indices 1-2 are eyes, 5-16 are body

# Color definitions
POSE_CONNECTIONS = [
    # Upper body (pink)
    ((2, 4), (255, 192, 203, LINE_OPACITY)),  # Left shoulder to left elbow
    ((4, 6), (255, 192, 203, LINE_OPACITY)),  # Left elbow to left wrist
    ((3, 5), (255, 192, 203, LINE_OPACITY)),  # Right shoulder to right elbow
    ((5, 7), (255, 192, 203, LINE_OPACITY)),  # Right elbow to right wrist
    
    # Lower body (orange)
    ((8, 10), (0, 127, 255, LINE_OPACITY)),  # Left hip to left knee
    ((10, 12), (0, 127, 255, LINE_OPACITY)),  # Left knee to left ankle
    ((9, 11), (0, 127, 255, LINE_OPACITY)),  # Right hip to right knee
    ((11, 13), (0, 127, 255, LINE_OPACITY)),  # Right knee to right ankle
    
    # Eyes (green)
    ((0, 1), (0, 255, 0, LINE_OPACITY)),    # Between eyes
]

POINT_COLORS = {
    'eyes': (0, 255, 0, POINT_OPACITY),      # Green
    'arms': (255, 192, 203, POINT_OPACITY),  # Pink
    'legs': (0, 127, 255, POINT_OPACITY)     # Orange
}

# Pre-compute color tuples
LINE_COLORS = {conn: color[:3] for (conn, color) in POSE_CONNECTIONS}
POINT_COLORS_RGB = {k: v[:3] for k, v in POINT_COLORS.items()}

def get_default_output_path(input_path: str, output_format: str = 'h264') -> str:
    """Generate default output path with appropriate extension based on format."""
    input_path = Path(input_path)
    ext = OUTPUT_FORMATS[output_format]['ext']
    return str(input_path.parent / f"{input_path.stem}_pose_detected{ext}")

def process_video_frames(
    model,
    input_path: str,
    debug: bool = False
) -> Generator[tuple[cv2.Mat, dict], None, None]:
    """
    Process video frames with YOLO pose detection.
    Returns frames with pose visualization overlay.
    """
    previous_points = None
    stats = TimingStats() if debug else None
    frame_idx = 0
    
    # Get video FPS from input
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    def get_best_detection(keypoints_list):
        """Return the detection with highest average confidence."""
        if keypoints_list is None or len(keypoints_list) == 0:
            return None
            
        best_idx = 0
        best_conf = 0
        
        for i, keypoints in enumerate(keypoints_list):
            if keypoints is None or len(keypoints) == 0:
                continue
                
            if len(keypoints) <= max(KEPT_KEYPOINTS):
                continue
                
            conf_values = keypoints[KEPT_KEYPOINTS][:, 2]
            if len(conf_values) == 0:
                continue
                
            conf = float(conf_values.mean())
            if conf > best_conf:
                best_conf = conf
                best_idx = i
        
        if best_conf == 0:
            return None
            
        return keypoints_list[best_idx]

    def smooth_points(current_points, prev_points, alpha):
        """Apply exponential smoothing to points."""
        if prev_points is None:
            return current_points
        
        smoothed = current_points.copy()
        
        mask = ((current_points[:, 2] > SMOOTHING_CONFIDENCE_THRESHOLD) & 
                (prev_points[:, 2] > SMOOTHING_CONFIDENCE_THRESHOLD))
        
        if np.sum(mask) < len(mask) // 2:
            return current_points
        
        smoothed[mask, 0] = (1 - alpha) * current_points[mask, 0] + alpha * prev_points[mask, 0]
        smoothed[mask, 1] = (1 - alpha) * current_points[mask, 1] + alpha * prev_points[mask, 1]
        
        basic_mask = ((current_points[:, 2] > CONFIDENCE_THRESHOLD) & ~mask)
        smoothed[basic_mask] = current_points[basic_mask]
        
        return smoothed

    for result in model.predict(
        source=input_path,
        stream=True,
        show=False,
        save=False,
        device=get_device(),
        verbose=False
    ):
        # Calculate timestamp in seconds
        timestamp = frame_idx / fps
        
        frame = result.orig_img
        keypoints_data = []
        
        if result.keypoints is not None:
            best_keypoints = get_best_detection(result.keypoints.data)
            if best_keypoints is not None:
                selected_points = best_keypoints[KEPT_KEYPOINTS]
                selected_points = selected_points.cpu().numpy()
                
                # Apply smoothing
                smoothed_points = smooth_points(selected_points, previous_points, SMOOTHING_FACTOR)
                previous_points = smoothed_points.copy()
                
                # Create keypoint dictionary for metadata
                keypoint_dict = {}
                for idx, name in enumerate(KEYPOINT_NAMES):
                    point = smoothed_points[idx]
                    keypoint_dict[name] = {
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "confidence": float(point[2])
                    }
                keypoints_data.append({"keypoints": keypoint_dict})
                
                # Draw visualization
                confidences = smoothed_points[:, 2]
                valid_mask = ((confidences > CONFIDENCE_THRESHOLD) & 
                             (smoothed_points[:, 0] > 1) &
                             (smoothed_points[:, 1] > 1) &
                             (smoothed_points[:, 0] < frame.shape[1]) &
                             (smoothed_points[:, 1] < frame.shape[0]))
                
                # Draw lines
                for (start_idx, end_idx), color in POSE_CONNECTIONS:
                    if (valid_mask[start_idx] and valid_mask[end_idx]):
                        start = smoothed_points[start_idx]
                        end = smoothed_points[end_idx]
                        
                        if (start[0] > 1 and start[1] > 1 and 
                            end[0] > 1 and end[1] > 1 and
                            start[0] < frame.shape[1] and start[1] < frame.shape[0] and
                            end[0] < frame.shape[1] and end[1] < frame.shape[0]):
                            cv2.line(frame,
                                    (int(start[0]), int(start[1])),
                                    (int(end[0]), int(end[1])),
                                    LINE_COLORS[(start_idx, end_idx)],
                                    LINE_WIDTH,
                                    cv2.LINE_AA)
                
                # Draw points
                valid_points = smoothed_points[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                for idx, point in zip(valid_indices, valid_points):
                    x, y = int(point[0]), int(point[1])
                    color = (POINT_COLORS_RGB['eyes'] if idx < 2 else 
                            POINT_COLORS_RGB['arms'] if idx < 8 else 
                            POINT_COLORS_RGB['legs'])
                    cv2.circle(frame, (x, y), POINT_RADIUS, color, -1, cv2.LINE_AA)

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "timestamp_ms": int(timestamp * 1000),
            "persons": keypoints_data
        }
        
        yield frame, frame_data
        frame_idx += 1

def create_video_writer(
    input_path: str, 
    output_path: str, 
    output_format: str = 'mjpeg'
) -> Union[cv2.VideoWriter, FFmpegWriter]:
    """Create a video writer with same properties as input video."""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Always use FFmpegWriter for MOV files or when output format is h264/webm
    if output_format in ['h264', 'webm'] or input_path.lower().endswith('.mov'):
        return FFmpegWriter(output_path, width, height, fps)
    
    # Try OpenCV VideoWriter first
    fourcc = OUTPUT_FORMATS[output_format]['fourcc']
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height)
    )
    
    # If OpenCV writer fails, fall back to FFmpegWriter
    if not writer.isOpened():
        print(f"Warning: {fourcc} codec not available, falling back to FFmpeg")
        return FFmpegWriter(output_path, width, height, fps)
        
    return writer

def process_video(
    input_path: str,
    model_path: str = None,
    output_path: Optional[str] = None,
    output_format: str = 'h264',
    debug: bool = False,
    skip_audio: bool = False  # New parameter
) -> None:
    """Process video with pose detection and audio syncing."""
    if debug:
        total_start = time.perf_counter()
        print("\nStarting video processing...")
        model_start = time.perf_counter()
    
    model = load_yolo_model(model_path)
            
    if debug:
        model_time = time.perf_counter() - model_start
        print(f"Model loading time: {model_time:.2f}s")
        writer_start = time.perf_counter()
    
    # Create temporary output path for video without audio
    temp_output = output_path or get_default_output_path(input_path, output_format)
    if not skip_audio and os.path.exists(input_path):
        base, ext = os.path.splitext(temp_output)
        temp_output = f"{base}_temp{ext}"
    
    writer = create_video_writer(input_path, temp_output, output_format)
    
    if debug:
        writer_init = time.perf_counter() - writer_start
        print(f"Video writer initialization: {writer_init:.2f}s")
        io_time = 0
        frame_count = 0
        
    try:
        for frame, _ in process_video_frames(model, input_path, debug=debug):
            if debug:
                write_start = time.perf_counter()
                
            writer.write(frame)
            
            if debug:
                io_time += time.perf_counter() - write_start
                frame_count += 1
    finally:
        writer.release()

    # Sync audio if input file exists and audio sync not skipped
    final_output = output_path or get_default_output_path(input_path, output_format)
    if not skip_audio and os.path.exists(input_path):
        if debug:
            print("\nSyncing audio...")
        try:
            sync_audio(input_path, temp_output, final_output)
            if temp_output != final_output:  # Only delete if paths are different
                os.remove(temp_output)  # Clean up temporary file
        except Exception as e:
            print(f"Warning: Audio sync failed: {str(e)}")
            print("Keeping video without audio")
            if temp_output != final_output:
                os.rename(temp_output, final_output)
    
    if debug:
        total_time = time.perf_counter() - total_start
        print("\nAdditional Timing Information:")
        print(f"Total wall time: {total_time:.2f}s")
        if frame_count > 0:
            print(f"Average I/O time per frame: {(io_time/frame_count)*1000:.2f}ms")
            print(f"Total I/O time: {io_time:.2f}s")
        print(f"Effective FPS (including all overhead): {frame_count/total_time:.1f}")
    
    print(f"Processed video saved to: {final_output}")

def main():
    parser = argparse.ArgumentParser(description="Process video with pose detection")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Path to YOLO model")
    parser.add_argument("--output", help="Output video path (default: input_pose_detected.mp4)")
    parser.add_argument("--output-format", 
                       choices=['mjpeg', 'h264', 'webm'],
                       default='h264',
                       help="Output video format (default: h264)")
    parser.add_argument("--debug", action="store_true", help="Output timing information")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio syncing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    process_video(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        output_format=args.output_format,
        debug=args.debug,
        skip_audio=args.skip_audio
    )

if __name__ == '__main__':
    main() 
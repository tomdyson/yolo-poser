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

Installation
-----------
pip install torch ultralytics opencv-python numpy

Usage
-----
Basic video processing:
    python pose-detect.py input_video.mp4

Arguments:
    input             Input video file path
    --model          Path to YOLO model (default: yolo11n-pose.pt)
    --output         Output video path (default: input_pose_detected.[avi/mp4/webm])
    --output-format  Output video format: 'mjpeg' or 'h264' or 'webm' (default: mjpeg)
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
- MJPEG (.avi): Default format, widely compatible
- H.264 (.mp4): More efficient compression, requires FFmpeg
- WebM (.webm): More efficient compression, requires FFmpeg
"""

import argparse
import os
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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

def get_device() -> torch.device:
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

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

def get_default_output_path(input_path: str, output_format: str = 'mjpeg') -> str:
    """Generate default output path with appropriate extension based on format."""
    input_path = Path(input_path)
    ext = OUTPUT_FORMATS[output_format]['ext']
    return str(input_path.parent / f"{input_path.stem}_pose_detected{ext}")

def process_video_frames(
    model: YOLO,
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

class FFmpegWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float):
        if output_path.endswith('.webm'):
            # WebM-specific FFmpeg command with VP9 codec
            command = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-c:v', 'libvpx-vp9',  # Use VP9 codec
                '-b:v', '2M',          # Target bitrate
                '-deadline', 'realtime',# Faster encoding
                '-cpu-used', '4',      # Speed/quality tradeoff (0-8, higher = faster)
                '-pix_fmt', 'yuv420p',
                '-f', 'webm',
                '-loglevel', 'error',
                output_path
            ]
        else:
            # Existing H.264 command
            command = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-profile:v', 'high',
                '-level:v', '4.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-f', 'mp4',
                '-loglevel', 'error',
                output_path
            ]
        
        # Redirect stderr to /dev/null to suppress remaining output
        self.process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
    def write(self, frame):
        self.process.stdin.write(frame.tobytes())
        
    def release(self):
        self.process.stdin.close()
        self.process.wait(timeout=5)

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
    
    if output_format in ['h264', 'webm']:
        return FFmpegWriter(output_path, width, height, fps)
    else:
        # Existing MJPEG writer code
        fourcc = OUTPUT_FORMATS[output_format]['fourcc']
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer. {fourcc} codec not available.")
        return writer

def process_video(
    input_path: str,
    model_path: str = None,  # Changed default to None
    output_path: Optional[str] = None,
    output_format: str = 'mjpeg',
    debug: bool = False
) -> None:
    
    if model_path is None:
        # Use the default model from the package
        model_path = "yolo11n-pose.pt"
        if not os.path.exists(model_path):
            print("Downloading YOLO model...")
            model = YOLO("yolo11n-pose.pt")  # This will download the model if it doesn't exist
    else:
        model = YOLO(model_path)
            
    if debug:
        total_start = time.perf_counter()
        print("\nStarting video processing...")
        model_start = time.perf_counter()
    
    if debug:
        model_time = time.perf_counter() - model_start
        print(f"Model {model_path} loading time: {model_time:.2f}s")
        writer_start = time.perf_counter()
    
    output_path = output_path or get_default_output_path(input_path, output_format)
    writer = create_video_writer(input_path, output_path, output_format)
    
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
        
    if debug:
        total_time = time.perf_counter() - total_start
        print("\nAdditional Timing Information:")
        print(f"Total wall time: {total_time:.2f}s")
        if frame_count > 0:
            print(f"Average I/O time per frame: {(io_time/frame_count)*1000:.2f}ms")
            print(f"Total I/O time: {io_time:.2f}s")
        print(f"Effective FPS (including all overhead): {frame_count/total_time:.1f}")
    
    print(f"Processed video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process video with pose detection")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Path to YOLO model")
    parser.add_argument("--output", help="Output video path (default: input_pose_detected.[avi/mp4/webm])")
    parser.add_argument("--output-format", 
                       choices=['mjpeg', 'h264', 'webm'],
                       default='mjpeg',
                       help="Output video format (default: mjpeg)")
    parser.add_argument("--debug", action="store_true", help="Output timing information")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    process_video(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        output_format=args.output_format,
        debug=args.debug
    )

if __name__ == "__main__":
    main() 
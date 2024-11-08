"""Shared utilities for YOLO-based video processing."""

import os
import subprocess
from pathlib import Path

import torch
from ultralytics import YOLO


def get_device() -> torch.device:
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_yolo_model(model_path: str = None) -> YOLO:
    """Load YOLO model, downloading default if not specified."""
    if model_path is None:
        # Use the default model from the package
        model_path = "yolo11n-pose.pt"
        if not os.path.exists(model_path):
            print("Downloading YOLO model...")
            model = YOLO("yolo11n-pose.pt")  # This will download if needed
        else:
            model = YOLO(model_path)
    else:
        model = YOLO(model_path)
    
    return model.to(get_device())

class FFmpegWriter:
    """FFmpeg-based video writer supporting multiple formats."""
    def __init__(self, output_path: str, width: int, height: int, fps: float):
        if output_path.endswith('.webm'):
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
                '-c:v', 'libvpx-vp9',
                '-b:v', '2M',
                '-deadline', 'realtime',
                '-cpu-used', '4',
                '-pix_fmt', 'yuv420p',
                '-f', 'webm',
                '-loglevel', 'error',
                output_path
            ]
        else:
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
                '-c:v', 'h264_videotoolbox',  # Use hardware acceleration if available
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
        
        self.process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
    def write(self, frame):
        """Write a frame to the video."""
        self.process.stdin.write(frame.tobytes())
        
    def release(self):
        """Close the video writer."""
        self.process.stdin.close()
        self.process.wait(timeout=5) 
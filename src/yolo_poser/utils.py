"""Shared utilities for YOLO-based video processing."""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
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

class FFmpegTools:
    """Utilities for FFmpeg operations including video writing and probing."""
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get the duration of a video file using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-print_format', 'json',
            '-show_entries', 'format=duration',
            '-select_streams', 'v:0',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                cmd = ['ffprobe', '-v', 'error', video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                raise ValueError(f"Could not read video file: {video_path}")
                
            data = json.loads(result.stdout)
            
            if 'format' in data and 'duration' in data['format']:
                return float(data['format']['duration'])
                
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-print_format', 'json',
                '-show_entries', 'stream=duration',
                '-select_streams', 'v:0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            data = json.loads(result.stdout)
            
            if 'streams' in data and data['streams'] and 'duration' in data['streams'][0]:
                return float(data['streams'][0]['duration'])
                
            raise ValueError(f"Could not determine duration for {video_path}")
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error getting duration for {video_path}: {str(e)}")
            print(f"ffprobe output: {result.stdout if 'result' in locals() else 'No output'}")
            raise

    @staticmethod
    def get_video_properties(video_path: str) -> Tuple[int, int, float]:
        """Get video width, height and fps."""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, fps

class FFmpegWriter:
    """FFmpeg-based video writer supporting multiple formats."""
    def __init__(self, output_path: str, width: int, height: int, fps: float):
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(f"Width and height must be even numbers. Got {width}x{height}")
        
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
                '-c:v', 'libx264',  # Changed back to libx264
                '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                output_path
            ]
        
        try:
            self.process = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to start FFmpeg: {str(e)}")
        
    def write(self, frame):
        """Write a frame to the video."""
        if self.process is None:
            raise RuntimeError("FFmpeg process is not initialized")
            
        if self.process.poll() is not None:
            stderr = self.process.stderr.read().decode() if self.process.stderr else "No error output"
            raise RuntimeError(f"FFmpeg process terminated unexpectedly\nFFmpeg error: {stderr}")
            
        try:
            self.process.stdin.write(frame.tobytes())
        except (IOError, BrokenPipeError) as e:
            stderr = self.process.stderr.read().decode() if self.process.stderr else "No error output"
            self.process.terminate()
            raise RuntimeError(f"FFmpeg write failed: {str(e)}\nFFmpeg error: {stderr}")
        
    def release(self):
        """Close the video writer."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=10)  # Increased timeout
                
                if self.process.returncode != 0:
                    stderr = self.process.stderr.read().decode() if self.process.stderr else "No error output"
                    raise RuntimeError(f"FFmpeg failed with code {self.process.returncode}: {stderr}")
                    
            except Exception as e:
                if self.process:
                    self.process.kill()
                raise RuntimeError(f"Error releasing FFmpeg writer: {str(e)}")
            finally:
                self.process = None
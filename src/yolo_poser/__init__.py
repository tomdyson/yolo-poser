"""YOLO-based human pose detection and visualization tool."""

__version__ = "0.1.17"

from .autocrop import calculate_crop_params, crop_video
from .pose_detect import process_video, process_video_frames
from .sync_audio import sync_audio

__all__ = [
    "process_video",
    "process_video_frames",
    "crop_video",
    "calculate_crop_params",
    "sync_audio"
] 
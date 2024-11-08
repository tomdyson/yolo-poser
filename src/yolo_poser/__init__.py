"""YOLO-based human pose detection and visualization tool."""

__version__ = "0.1.4"

from .pose_detect import process_video, process_video_frames

__all__ = ["process_video", "process_video_frames"] 
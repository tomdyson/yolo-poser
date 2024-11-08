# YOLO Poser

A Python package for human pose detection and visualization using YOLO.

## Installation 

```bash
pip install yolo-poser
```

## Usage

### Command Line
```bash
yolo-poser input_video.mp4 --output output.mp4 --output-format h264
```

### Python API

```python
from yolo_poser import process_video
process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    output_format="h264",
    debug=True
)
```

## Features

- Human pose detection using YOLO
- Support for multiple output formats (MJPEG, H264, WebM)
- Smooth keypoint tracking
- Debug mode with performance metrics
- Configurable visualization options

## Requirements

- Python 3.8+ (<3.13)
- PyTorch
- Ultralytics
- OpenCV
- NumPy

## License

MIT License
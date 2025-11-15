# YOLO Poser

A Python package for human pose detection and visualization using YOLO.

## Installation 

```bash
pip install yolo-poser
```

## Usage

### Pose Detection

Process videos to detect and visualize human poses.

**Command Line:**
```bash
yolo-poser input_video.mp4 --output output.mp4 --output-format h264
```

**Python API:**
```python
from yolo_poser import process_video
process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    output_format="h264",
    debug=True
)
```

### Video Cropping

Automatically crop videos to focus on detected people using YOLO person detection.

**Command Line:**
```bash
yolo-crop input_video.mp4 --padding 0.3 --output cropped.mp4
```

Options:
- `--padding`: Padding around detected area (default: 0.5)
- `--no-keep-proportions`: Don't maintain original video proportions
- `--no-preview`: Skip preview and confirmation
- `--debug`: Save debug frames showing detections
- `--output`: Output video path (default: input_cropped.mp4)

**Python API:**
```python
from yolo_poser import crop_video, calculate_crop_params

# Calculate crop parameters
bbox = (100, 100, 500, 700)  # min_x, min_y, max_x, max_y
crop_params = calculate_crop_params(
    video_path="input.mp4",
    bbox=bbox,
    padding=0.3,
    keep_proportions=True
)

# Apply crop
crop_video(
    input_path="input.mp4",
    output_path="cropped.mp4",
    crop_params=crop_params
)
```

### Video Splitting

Split videos into shots based on audio peaks. Useful for creating training data or breaking down performances.

**Command Line:**
```bash
yolo-split input.mp4 --output-dir shots/ --peak-sensitivity 0.5 --shot-duration 2.5
```

Options:
- `--output-dir`: Directory for output files (default: same as input)
- `--peak-sensitivity`: Peak detection sensitivity, 0.1-1.0 (default: 0.8). Lower values create more shots
- `--shot-duration`: Duration of each shot in seconds (default: 2.0)
- `--debug`: Keep temporary audio files
- `--json`: Output JSON list of generated files to stdout

**Python API:**
```python
from yolo_poser.split import split_video

result = split_video(
    input_path="input.mp4",
    output_dir="shots/",
    peak_sensitivity=0.8,
    shot_duration=2.0
)
# Returns: {'chunks': ['shots/chunk_001.mp4', 'shots/chunk_002.mp4', ...]}
```

### Audio Syncing

Sync audio from one video to another, with automatic duration adjustment.

**Python API:**
```python
from yolo_poser import sync_audio

output_path = sync_audio(
    source_video="original.mp4",    # Video with audio
    destination_video="processed.mp4",  # Video without audio
    output_path="final.mp4"  # Optional
)
```

### Web API

To use the HTTP API, first install with API dependencies:

```bash
pip install "yolo-poser[api]"
```

Start the API server:
```bash
yolo-poser-api [--host HOST] [--port PORT]
```

For example:
```bash
yolo-poser-api --host 127.0.0.1 --port 9000
```

Or programmatically:
```python
from yolo_poser.api import app
import uvicorn

uvicorn.run(app, host="127.0.0.1", port=9000)
```

The API provides endpoints for:
- Processing videos from URLs: POST /detect/url
- Processing uploaded video files: POST /detect/file
- Health check: GET /health

See the API documentation at http://localhost:8000/docs when running the server.

## Features

- **Pose Detection**: Human pose detection and visualization using YOLO
- **Video Cropping**: Automatically crop videos to focus on detected people
- **Video Splitting**: Split videos into shots based on audio peaks
- **Audio Syncing**: Sync audio from one video to another with automatic duration adjustment
- **Multiple Output Formats**: Support for MJPEG, H264, and WebM
- **Smooth Tracking**: Exponential smoothing for stable keypoint tracking
- **HTTP API**: FastAPI-based REST API for video processing
- **Debug Mode**: Performance metrics and visualization of detection results

## Requirements

- Python 3.8+ (<3.13)
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- SciPy (for video splitting)
- FFmpeg (for video processing and audio syncing)

## Development

### Continuous Integration

This project uses GitHub Actions for continuous integration and deployment:

- Every push to the `main` branch triggers a test build that publishes to TestPyPI
- Tagged releases (e.g. `v0.1.0`) trigger a build that publishes to PyPI

To release a new version:

1. Update the version in `src/yolo_poser/__init__.py`
2. Commit the changes
3. Create and push a tag:
```bash
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Action will automatically build and publish the new version to PyPI.

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/tomdyson/yolo-poser.git
cd yolo-poser
```

2. Install in development mode:
```bash
pip install -e .
```

## License

MIT License
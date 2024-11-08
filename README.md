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
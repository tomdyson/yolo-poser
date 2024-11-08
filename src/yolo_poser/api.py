#!/usr/bin/env python3

import argparse
import os
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from starlette.background import BackgroundTask

from yolo_poser import process_video

app = FastAPI(
    title="Pose Detection API",
    description="API for detecting human poses in videos using YOLO",
    version="1.0.0"
)

# Create a temporary directory for storing uploads and processed videos
TEMP_DIR = Path(tempfile.gettempdir()) / "pose_detection"
TEMP_DIR.mkdir(exist_ok=True)

class URLInput(BaseModel):
    url: HttpUrl
    model: Optional[str] = "yolo11n-pose.pt"
    output_format: Optional[str] = "mjpeg"

@app.post("/detect/url", 
         summary="Detect poses in video from URL",
         description="Process a video from a URL and return the processed video with pose detection")
async def detect_from_url(input_data: URLInput):
    input_path = None
    output_path = None
    
    try:
        # Generate unique filenames for input and output
        input_filename = f"{uuid.uuid4()}_input.mp4"
        output_filename = f"{uuid.uuid4()}_output{'.avi' if input_data.output_format == 'mjpeg' else '.mp4'}"
        
        input_path = TEMP_DIR / input_filename
        output_path = TEMP_DIR / output_filename
        
        # Download the video
        urllib.request.urlretrieve(str(input_data.url), input_path)
        
        # Process the video using yolo-poser
        process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            output_format=input_data.output_format,
            model_path=input_data.model_path
        )
        
        # Clean up input file
        if input_path.exists():
            input_path.unlink()
        
        # Return the processed video
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="video/mp4" if input_data.output_format == "h264" else "video/x-msvideo",
            background=BackgroundTask(output_path.unlink)
        )
        
    except Exception as e:
        # Clean up files in case of error
        if input_path and input_path.exists():
            input_path.unlink()
        if output_path and output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/file",
          summary="Detect poses in uploaded video",
          description="Process an uploaded video file and return the processed video with pose detection")
async def detect_from_file(
    file: UploadFile = File(...),
    model: str = Form("yolo11n-pose.pt"),
    output_format: str = Form("mjpeg")
):
    input_path = None
    output_path = None
    
    try:
        # Generate unique filenames for input and output
        input_filename = f"{uuid.uuid4()}_{file.filename}"
        output_filename = f"{uuid.uuid4()}_output{'.avi' if output_format == 'mjpeg' else '.mp4'}"
        
        input_path = TEMP_DIR / input_filename
        output_path = TEMP_DIR / output_filename
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the video using yolo-poser
        process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            output_format=output_format,
            model_path=model
        )
        
        # Clean up input file
        if input_path.exists():
            input_path.unlink()
        
        # Return the processed video
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="video/mp4" if output_format == "h264" else "video/x-msvideo",
            background=BackgroundTask(output_path.unlink)
        )
        
    except Exception as e:
        # Clean up files in case of error
        if input_path and input_path.exists():
            input_path.unlink()
        if output_path and output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health",
         summary="Health check",
         description="Check if the API is running")
async def health_check():
    return {"status": "healthy"}

def main():
    """Entry point for the API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: API dependencies not installed. Please install with:")
        print("pip install 'yolo-poser[api]'")
        return

    parser = argparse.ArgumentParser(description="Start the YOLO Poser API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    args = parser.parse_args()

    print(f"Starting YOLO Poser API server on {args.host}:{args.port}...")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 
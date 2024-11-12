#!/usr/bin/env python3

"""
Video Splitting Tool
==================

Splits a video into shots based on audio peaks. Useful for creating training
data from longer videos, or breaking down performances into individual shots.

Features:
- Detects audio peaks to identify natural break points
- Creates fixed-duration shots with peaks at consistent positions
- Adds shot numbers as visual overlays
- Outputs clean JSON for automation

Usage:
------
Basic usage:
    yolo-split input.mp4

With options:
    yolo-split input.mp4 --output-dir shots/ --peak-sensitivity 0.5 --shot-duration 2.5

Parameters:
----------
peak_sensitivity : float (0.1-1.0)
    How sensitive the peak detection is. Lower values create more shots.
    Default: 0.8

shot_duration : float
    Duration of each output shot in seconds.
    Default: 2.0

Output:
-------
- Creates numbered video files (chunk_001.mp4, chunk_002.mp4, etc.)
- Each shot has its number burned into the top-left corner
- With --json flag, outputs a list of generated filenames

Example JSON output:
{
    "chunks": [
        "output/chunk_001.mp4",
        "output/chunk_002.mp4"
    ]
}

Requirements:
------------
- FFmpeg for video processing
- scipy for audio analysis
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks


def extract_audio(video_file: Path, output_file: Path, quiet: bool = False) -> Path:
    """Extract audio from video file"""
    if not quiet:
        print("Extracting audio...")
    
    # First check if input file exists
    if not video_file.exists():
        raise FileNotFoundError(f"Input video file not found: {video_file}")
        
    cmd = [
        'ffmpeg',
        '-i', str(video_file),
        '-ab', '160k',
        '-ac', '2',
        '-ar', '44100',
        '-vn',
        '-y',  # Overwrite output file
        str(output_file)
    ]
    
    # Add quiet flags to ffmpeg when in quiet mode
    if quiet:
        cmd.insert(1, '-hide_banner')
        cmd.insert(2, '-loglevel')
        cmd.insert(3, 'error')
        
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Get ffmpeg's error message from stderr
        error_msg = e.stderr.strip() if e.stderr else "Unknown error"
        raise RuntimeError(f"Failed to extract audio from {video_file}. FFmpeg error: {error_msg}")
        
    return output_file

def find_audio_peaks(audio_file: Path, height: float = 0.8, distance: int = 44100, quiet: bool = False) -> np.ndarray:
    """Find the time positions of audio peaks"""
    if not quiet:
        print("Analyzing audio for peaks...")
    rate, data = wavfile.read(audio_file)
    # If stereo, take the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    peaks, _ = find_peaks(data, height=height * max(data), distance=distance)
    times = peaks / rate
    return times

def split_chunks(
    input_file: Path,
    times: np.ndarray,
    output_dir: Path,
    chunk_duration: float = 2.0,
    peak_position: float = 0.4,
    quiet: bool = False
) -> List[Path]:
    """Split video into chunks at peak positions"""
    if not quiet:
        print(f"Splitting video into {chunk_duration}-second chunks (peak at {peak_position*100}% position)...")
    chunks = []

    for i, time in enumerate(times, start=1):
        chunk_file = output_dir / f"chunk_{i:03d}.mp4"
        chunks.append(chunk_file)

        # Calculate start time so peak occurs at desired position
        start_time = max(time - (peak_position * chunk_duration), 0)
        if not quiet:
            print(f"Processing chunk {i}/{len(times)} (starting at {start_time:.2f}s)...")

        # Define drawtext filter for the chapter number
        drawtext_filter = (
            f"drawtext=text='{i}'"
            ":fontcolor=white"
            ":fontsize=h/6"
            ":box=1"
            ":boxcolor=black@0.5"
            ":boxborderw=5"
            ":x=10"
            ":y=10"
            ":font=Arial"
        )

        cmd = [
            'ffmpeg',
            '-ss', f'{start_time}',
            '-i', str(input_file),
            '-t', f'{chunk_duration}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-profile:v', 'high',
            '-level:v', '4.1',
            '-pix_fmt', 'yuv420p',
            '-vf', drawtext_filter,
            '-force_key_frames', 'expr:gte(t,0)',
            '-movflags', '+faststart',
            '-video_track_timescale', '90000',
            '-c:a', 'aac',
            '-b:a', '256k',
            '-ar', '48000',
            '-ac', '2',
            '-avoid_negative_ts', '1',
            '-y',
            str(chunk_file)
        ]

        # Add quiet flags to ffmpeg when in quiet mode
        if quiet:
            cmd.insert(1, '-hide_banner')
            cmd.insert(2, '-loglevel')
            cmd.insert(3, 'error')

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create chunk {i}: {e.stderr}")

    return chunks

def split_video(
    input_path: str,
    output_dir: Path = None,
    peak_sensitivity: float = 0.8,
    shot_duration: float = 2.0,
    debug: bool = False,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Split video into shots based on audio peaks.
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save output files (default: same as input)
        peak_sensitivity: How sensitive peak detection is (0.1-1.0, default: 0.8)
        shot_duration: Duration of each shot in seconds (default: 2.0)
        debug: Enable debug output
        quiet: Suppress all output except JSON
        
    Returns:
        Dictionary containing:
            chunks: List of paths to generated chunks
    """
    # Set up paths
    input_file = Path(input_path)
    if output_dir is None:
        output_dir = input_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio
    audio_file = output_dir / f"audio-{input_file.name}.wav"
    if not audio_file.exists():
        extract_audio(input_file, audio_file, quiet)

    # Find peaks in audio
    peaks = find_audio_peaks(audio_file, height=peak_sensitivity, quiet=quiet)
    if not quiet:
        print(f"Found {len(peaks)} peaks")

    # Split video into chunks
    chunks = split_chunks(
        input_file=input_file,
        times=peaks,
        output_dir=output_dir,
        chunk_duration=shot_duration,
        quiet=quiet
    )

    # Clean up audio file if not in debug mode
    if not debug and audio_file.exists():
        audio_file.unlink()

    # Return results - only chunks, no peaks
    return {
        'chunks': [str(chunk) for chunk in chunks]
    }

def main():
    parser = argparse.ArgumentParser(description="Split video into shots based on audio peaks")
    parser.add_argument("input", help="Input video file")
    parser.add_argument(
        "--output-dir", 
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--peak-sensitivity",
        type=float,
        default=0.8,
        help="How sensitive peak detection is (0.1-1.0, default: 0.8). Lower values create more shots."
    )
    parser.add_argument(
        "--shot-duration",
        type=float,
        default=2.0,
        help="Duration of each shot in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug output and keep temporary files"
    )
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output JSON to stdout (suppresses other output)"
    )
    
    args = parser.parse_args()
    
    try:
        result = split_video(
            input_path=args.input,
            output_dir=args.output_dir,
            peak_sensitivity=args.peak_sensitivity,
            shot_duration=args.shot_duration,
            debug=args.debug,
            quiet=args.json  # Silence output when JSON is requested
        )
        
        if args.json:
            print(json.dumps(result))
        else:
            print("\nGenerated shot files:")
            for chunk in result['chunks']:
                print(f"  {chunk}")
            print(f"\nTotal shots: {len(result['chunks'])}")
            
    except Exception as e:
        if not args.json:
            print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main()

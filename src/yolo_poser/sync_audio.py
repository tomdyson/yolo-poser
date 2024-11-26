#!/usr/bin/env python3
import argparse
import os
import subprocess

from .utils import FFmpegTools


def sync_audio(source_video, destination_video, output_path=None):
    """
    Extract audio from source video and add it to destination video,
    stretching/compressing if durations don't match.
    """
    if output_path is None:
        output_path = destination_video.replace('.', '_with_audio.')

    # Get durations using FFmpegTools
    source_duration = FFmpegTools.get_video_duration(source_video)
    dest_duration = FFmpegTools.get_video_duration(destination_video)

    # Extract audio from source
    temp_audio = source_video + '.temp_audio.aac'
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    extract_cmd = [
        'ffmpeg',
        '-v', 'quiet',
        '-i', source_video,
        '-vn',  # No video
        '-acodec', 'copy',  # Copy audio codec
        temp_audio
    ]
    subprocess.run(extract_cmd, check=True)

    # If durations are different, we need to stretch/compress audio
    if abs(source_duration - dest_duration) > 0.1:  # 0.1s tolerance
        tempo = source_duration / dest_duration
        # Use atempo filter to adjust audio speed
        # Note: atempo filter is limited to 0.5-2.0 range
        filter_complex = f'atempo={tempo}'
        if tempo > 2.0 or tempo < 0.5:
            # For larger changes, chain multiple atempo filters
            if tempo > 2.0:
                steps = [2.0] * (int(tempo/2))
                remainder = tempo/2 - int(tempo/2)
                if remainder > 0:
                    steps.append(1 + remainder)
            else:  # tempo < 0.5
                steps = [0.5] * (int(2/tempo))
                remainder = 2/tempo - int(2/tempo)
                if remainder > 0:
                    steps.append(1 - remainder)
            filter_complex = ','.join([f'atempo={step}' for step in steps])

        # Create adjusted audio
        temp_adjusted = 'temp_adjusted.aac'
        adjust_cmd = [
            'ffmpeg',
            '-v', 'quiet',
            '-i', temp_audio,
            '-filter:a', filter_complex,
            temp_adjusted
        ]
        subprocess.run(adjust_cmd, check=True)
        audio_input = temp_adjusted
    else:
        audio_input = temp_audio

    # Combine video and audio
    combine_cmd = [
        'ffmpeg',
        '-v', 'quiet',
        '-i', destination_video,  # Video input
        '-i', audio_input,        # Audio input
        '-c:v', 'copy',           # Copy video codec
        '-c:a', 'aac',            # AAC audio codec
        '-strict', 'experimental',
        output_path
    ]
    subprocess.run(combine_cmd, check=True)

    # Clean up temporary files
    subprocess.run(['rm', temp_audio])
    if audio_input != temp_audio:
        subprocess.run(['rm', temp_adjusted])

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Sync audio from source video to destination video')
    parser.add_argument('--source-video', required=True, help='Source video file (with audio)')
    parser.add_argument('--destination-video', required=True, help='Destination video file (without audio)')
    parser.add_argument('--output', help='Output video file (optional)')
    
    args = parser.parse_args()
    
    output_path = sync_audio(
        args.source_video,
        args.destination_video,
        args.output
    )
    print(f"Video with synced audio saved to: {output_path}")

if __name__ == "__main__":
    main() 
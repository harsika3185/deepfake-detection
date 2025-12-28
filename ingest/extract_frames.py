import os
import sys
import ffmpeg


def extract_frames(video_path, output_dir, fps=3):
    """
    Extract frames from a video using FFmpeg.

    Args:
        video_path (str): path to input video
        output_dir (str): directory to save frames
        fps (int): frames per second to extract
    """
    os.makedirs(output_dir, exist_ok=True)

    (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .output(
            os.path.join(output_dir, "frame_%05d.jpg"),
            qscale=2,
            start_number=0
        )
        .overwrite_output()
        .run(quiet=True)
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_frames.py <video_path> <output_dir>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Extracting frames from: {video_path}")
    extract_frames(video_path, output_dir)
    print(f"Frames saved to: {output_dir}")

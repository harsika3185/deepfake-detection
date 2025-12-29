import os
import sys
import ffmpeg


def extract_audio(video_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    (
        ffmpeg
        .input(video_path)
        .output(
            output_path,
            ac=1,          # mono
            ar=16000,      # 16 kHz
            format="wav"
        )
        .overwrite_output()
        .run(quiet=True)
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_audio.py <video_path> <output_audio.wav>")
        sys.exit(1)

    extract_audio(sys.argv[1], sys.argv[2])
    print("Audio extracted successfully")

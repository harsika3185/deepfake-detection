import os
import sys
from PIL import Image
from facenet_pytorch import MTCNN
import torch


def extract_faces(frames_dir, faces_dir, image_size=224):
    """
    Detect and crop faces from frames using MTCNN.

    Args:
        frames_dir (str): directory with extracted frames
        faces_dir (str): directory to save cropped face images
        image_size (int): output face size
    """
    os.makedirs(faces_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=image_size, margin=20, device=device)

    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(".jpg")
    ])

    print(f"Processing {len(frame_files)} frames...")

    saved = 0
    for idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_name)

        try:
            img = Image.open(frame_path).convert("RGB")
            face = mtcnn(img)

            if face is not None:
                face_img = Image.fromarray(
                    (face.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                out_path = os.path.join(faces_dir, f"face_{saved:05d}.jpg")
                face_img.save(out_path)
                saved += 1

        except Exception as e:
            print(f"Skipping {frame_name}: {e}")

    print(f"Saved {saved} face images to {faces_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python face_track.py <frames_dir> <faces_dir>")
        sys.exit(1)

    frames_dir = sys.argv[1]
    faces_dir = sys.argv[2]

    extract_faces(frames_dir, faces_dir)

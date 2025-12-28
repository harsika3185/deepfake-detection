import torch
from torchvision import transforms
from PIL import Image
import os

from pretrained_visual import PretrainedVisualDetector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = PretrainedVisualDetector().to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    faces_dir = "outputs/faces"
    face_files = sorted([
        f for f in os.listdir(faces_dir)
        if f.lower().endswith(".jpg")
    ])

    if len(face_files) == 0:
        print("No face images found!")
        return

    # Load first face image
    face_path = os.path.join(faces_dir, face_files[0])
    img = Image.open(face_path).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    print(f"Fake probability: {prob:.4f}")
    print(f"Face used: {face_path}")


if __name__ == "__main__":
    main()

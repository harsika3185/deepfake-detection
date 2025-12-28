import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from pretrained_visual import PretrainedVisualDetector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PretrainedVisualDetector().to(device)
    model.eval()

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

    if not face_files:
        print("No face images found.")
        return

    scores = []

    with torch.no_grad():
        for f in face_files:
            img = Image.open(os.path.join(faces_dir, f)).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).item()
            scores.append(prob)

    scores = np.array(scores)

    print("Frames analyzed:", len(scores))
    print("Mean fake probability   :", round(scores.mean(), 4))
    print("Median fake probability :", round(np.median(scores), 4))
    print("Std deviation           :", round(scores.std(), 4))

    final_score = np.median(scores)
    decision = "FAKE" if final_score > 0.5 else "REAL"

    print("\nFINAL DECISION:", decision)


if __name__ == "__main__":
    main()

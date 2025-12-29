import torch
import soundfile as sf

from pretrained_audio import PretrainedAudioDetector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PretrainedAudioDetector().to(device)
    model.eval()

    waveform, sr = sf.read("outputs/audio.wav")

    if sr != 16000:
        raise ValueError("Audio must be 16kHz")

    waveform = torch.tensor(
        waveform,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(waveform)
        prob = torch.sigmoid(logits).item()

    print(f"Audio fake probability: {prob:.4f}")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torchaudio


class PretrainedAudioDetector(nn.Module):
    def __init__(self):
        super().__init__()

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = bundle.get_model()

        # Freeze backbone
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        x: waveform tensor [1, T]
        """
        with torch.no_grad():
            features, _ = self.wav2vec.extract_features(x)

        pooled = features[-1].mean(dim=1)
        return self.classifier(pooled)

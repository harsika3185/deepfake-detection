import torch
import torch.nn as nn
import timm


class PretrainedVisualDetector(nn.Module):
    """
    Pretrained EfficientNet-based visual deepfake detector.
    """

    def __init__(self):
        super().__init__()

        # Load EfficientNet-B0 pretrained on ImageNet
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0  # remove classification head
        )

        # Simple classification head for fake/real
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from .cbam import CBAM

class EfficientNetB1_CBAM(nn.Module):
    """
    EfficientNet-B1 augmented with CBAM Dual-Attention Module.
    Primary model recommended in the review paper.
    Parameters: ~7.8M (lightweight and optimal for medical imaging).
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        base = efficientnet_b1(weights=weights)
        
        # Backbone feature extractor (ends with 1280 channels)
        self.features = base.features
        in_channels = 1280
        
        # CBAM Attention block placed right after features
        self.cbam = CBAM(in_channels=in_channels, reduction_ratio=16, kernel_size=7)
        
        # Global pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        attended_feat = self.cbam(feat)
        pooled = self.avgpool(attended_feat)
        flat = torch.flatten(pooled, 1)
        out = self.classifier(flat)
        return out


def build_efficientnet_cbam(num_classes: int = 3, pretrained: bool = True) -> EfficientNetB1_CBAM:
    return EfficientNetB1_CBAM(num_classes=num_classes, pretrained=pretrained)

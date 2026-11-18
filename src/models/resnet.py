import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50_Baseline(nn.Module):
    """
    Standard ResNet50 baseline model for comparative evaluation.
    Parameters: ~25.5M.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        
        # Keep conv layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        
        in_features = base.fc.in_features  # 2048
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet50(num_classes: int = 3, pretrained: bool = True) -> ResNet50_Baseline:
    return ResNet50_Baseline(num_classes=num_classes, pretrained=pretrained)

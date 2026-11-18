import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Equation 7 in paper):
    M_c(F) = sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    
    Tells the network *WHAT* features to focus on by assessing
    inter-channel relationships.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        weights = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return x * weights


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (Equation 8 in paper):
    M_s(F) = sigma(f^{7x7}([AvgPool(F); MaxPool(F)]))
    
    Tells the network *WHERE* to focus across the 2D slice
    by amplifying nodule regions and ignoring healthy tissue.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        weights = self.sigmoid(scale)
        return x * weights


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Sequentially applies Channel Attention followed by Spatial Attention.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

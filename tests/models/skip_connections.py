"""Skip connection test models (residual, dense)."""

import torch
import torch.nn as nn

from .base import register_model


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # Skip connection
        out = self.relu(out)
        return out


@register_model(
    name="DeepResNet",
    input_shapes=[(3, 32, 32)],
    categories=["skip_connections"],
    description="Deep ResNet with multiple residual blocks",
)
class DeepResNet(nn.Module):
    """Deep ResNet with multiple residual blocks.

    Architecture:
        - Initial conv layer
        - 3 residual blocks
        - Global average pooling
        - Classification head
    """

    def __init__(self, num_classes: int = 10, base_channels: int = 32):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU()

        # Residual blocks
        self.res_block1 = ResidualBlock(base_channels)
        self.res_block2 = ResidualBlock(base_channels)
        self.res_block3 = ResidualBlock(base_channels)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DenseLayer(nn.Module):
    """Single dense layer that concatenates input with output."""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)  # Dense connection


@register_model(
    name="DenseBlock",
    input_shapes=[(3, 32, 32)],
    categories=["skip_connections"],
    description="DenseNet-style block with concatenation-based connections",
)
class DenseBlock(nn.Module):
    """DenseNet-style model with dense connections.

    Architecture:
        - Initial conv
        - Dense block with 4 layers (each concatenates with all previous)
        - Transition layer
        - Classification head
    """

    def __init__(self, num_classes: int = 10, growth_rate: int = 16):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # Dense block: 4 layers with growth_rate
        # Channels: 32 -> 48 -> 64 -> 80 -> 96
        self.dense1 = DenseLayer(32, growth_rate)
        self.dense2 = DenseLayer(32 + growth_rate, growth_rate)
        self.dense3 = DenseLayer(32 + 2 * growth_rate, growth_rate)
        self.dense4 = DenseLayer(32 + 3 * growth_rate, growth_rate)

        final_channels = 32 + 4 * growth_rate  # 96

        # Transition layer
        self.transition = nn.Sequential(
            nn.BatchNorm2d(final_channels),
            nn.ReLU(),
            nn.Conv2d(final_channels, final_channels // 2, 1),
            nn.AvgPool2d(2),
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(final_channels // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.transition(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

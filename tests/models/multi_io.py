"""Multi-input/output test models."""

import torch
import torch.nn as nn

from .base import register_model


@register_model(
    name="SiameseEncoder",
    input_shapes=[(3, 64, 64), (3, 64, 64)],
    categories=["multi_io", "shared_weights"],
    description="Siamese network: applies same encoder to two images"
)
class SiameseEncoder(nn.Module):
    """Siamese encoder that applies the same CNN to two inputs.

    Architecture:
        - Shared CNN encoder for both inputs
        - Returns two embeddings
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            x1: First image (B, 3, 64, 64)
            x2: Second image (B, 3, 64, 64)

        Returns:
            Tuple of two embeddings (B, embed_dim) each
        """
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        return emb1, emb2


@register_model(
    name="MultiTaskHead",
    input_shapes=[(3, 32, 32)],
    categories=["multi_io"],
    description="Shared backbone with multiple output heads"
)
class MultiTaskHead(nn.Module):
    """Multi-task model with shared backbone and multiple heads.

    Architecture:
        - Shared CNN backbone
        - Classification head (10 classes)
        - Auxiliary head (regression)
    """

    def __init__(self, num_classes: int = 10, aux_dim: int = 4):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        # Auxiliary head (e.g., bounding box regression)
        self.aux_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, aux_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            x: Input image (B, 3, 32, 32)

        Returns:
            Tuple of (classification logits, auxiliary output)
        """
        features = self.backbone(x)
        cls_out = self.classifier(features)
        aux_out = self.aux_head(features)
        return cls_out, aux_out

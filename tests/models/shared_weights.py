"""Shared weights test models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import register_model


@register_model(
    name="RecurrentUnroll",
    input_shapes=[(32,)],
    categories=["shared_weights"],
    description="Same linear layer applied 5 times (unrolled RNN-style)"
)
class RecurrentUnroll(nn.Module):
    """Unrolled recurrent-style model with shared weights.

    Architecture:
        - Single Linear layer applied 5 times
        - Simulates RNN cell unrolling
    """

    def __init__(self, hidden_dim: int = 32, num_steps: int = 5):
        super().__init__()
        self.num_steps = num_steps
        # Shared cell
        self.cell = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, hidden_dim)

        Returns:
            Output after num_steps applications (B, hidden_dim)
        """
        h = x
        for _ in range(self.num_steps):
            h = self.activation(self.cell(h))
        return h


@register_model(
    name="WeightTying",
    input_shapes=[(16,)],
    categories=["shared_weights"],
    description="Embedding layer with tied input/output weights"
)
class WeightTying(nn.Module):
    """Model with tied embedding and output weights.

    Architecture:
        - Embedding layer
        - Transform layer
        - Output uses transposed embedding weights

    Common in language models (e.g., GPT).
    """

    def __init__(self, vocab_size: int = 100, embed_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transform
        self.transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Output bias (weight is tied to embedding)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input token indices (B, seq_len)

        Returns:
            Logits over vocabulary (B, seq_len, vocab_size)
        """
        # Embed
        embedded = self.embedding(x)  # (B, seq_len, embed_dim)

        # Transform
        hidden = self.transform(embedded)  # (B, seq_len, embed_dim)

        # Output with tied weights
        # hidden @ embedding.weight.T + bias
        logits = F.linear(hidden, self.embedding.weight, self.output_bias)

        return logits

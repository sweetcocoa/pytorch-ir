"""Attention and transformer test models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import register_model


@register_model(
    name="SelfAttention",
    input_shapes=[(16, 64)],
    categories=["attention"],
    description="Basic multi-head self-attention (4 heads)",
)
class SelfAttention(nn.Module):
    """Basic multi-head self-attention.

    Architecture:
        - Linear projections for Q, K, V
        - Scaled dot-product attention
        - Output projection
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, seq_len, d_model)

        Returns:
            Output tensor (B, seq_len, d_model)
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, L, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # (B, H, L, D)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


@register_model(
    name="CrossAttention",
    input_shapes=[(16, 64), (32, 64)],
    categories=["attention", "multi_io"],
    description="Cross-attention with query and context inputs",
)
class CrossAttention(nn.Module):
    """Cross-attention module.

    Architecture:
        - Query from one input, Key/Value from another
        - Used in encoder-decoder architectures
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q projection (from query input)
        self.q_proj = nn.Linear(d_model, d_model)
        # K, V projections (from context input)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query input (B, query_len, d_model)
            context: Context input (B, context_len, d_model)

        Returns:
            Output tensor (B, query_len, d_model)
        """
        B, Q_L, _ = query.shape
        _, C_L, _ = context.shape

        # Project
        q = self.q_proj(query)  # (B, Q_L, d_model)
        k = self.k_proj(context)  # (B, C_L, d_model)
        v = self.v_proj(context)

        # Reshape for multi-head
        q = q.view(B, Q_L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Q_L, D)
        k = k.view(B, C_L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, C_L, D)
        v = v.view(B, C_L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, Q_L, C_L)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # (B, H, Q_L, D)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q_L, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """Feed-forward network for transformer."""

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


@register_model(
    name="TransformerBlock",
    input_shapes=[(16, 64)],
    categories=["attention", "skip_connections"],
    description="Complete transformer block with attention, FFN, and skip connections",
)
class TransformerBlock(nn.Module):
    """Complete transformer encoder block.

    Architecture:
        - Multi-head self-attention
        - Add & LayerNorm (residual connection)
        - Feed-forward network
        - Add & LayerNorm (residual connection)
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = None):
        super().__init__()

        # Self-attention
        self.self_attn = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, seq_len, d_model)

        Returns:
            Output tensor (B, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

"""
Transformer Architecture Building Blocks.
Implements Pre-LayerNorm, RMSNorm, Feed-Forward Network (MLP), and Transformer Blocks.
"""

import torch
import torch.nn as nn
from .config import LLMConfig
from .attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Scales inputs by root mean square, reducing computational cost compared to standard LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP).
    Expands representation by 4x, applies non-linear activation (GELU), and projects back.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Decoder-Only Transformer Block using Pre-Normalization design:
    x = x + Attention(Norm1(x))
    x = x + MLP(Norm2(x))
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        if config.norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
            self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=True)

        self.attn = CausalSelfAttention(config)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN with residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

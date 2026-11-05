"""
Causal Multi-Head Self-Attention Implementation.
Supports both PyTorch 2.0+ Fast Scaled Dot-Product Attention (FlashAttention)
and transparent mathematical first-principles manual attention with causal masking.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import LLMConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as used in modern LLMs (LLaMA, Mistral)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # Shape of cos/sin: (seq_len, dim) -> broadcast to (B, H, T, D)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention module.
    Enforces autoregressive property: token at index i can only attend to tokens at indices <= i.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # Batched Q, K, V linear projection in a single efficient matrix multiply
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask for manual attention (lower triangular boolean matrix)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False
        )

        # Optional RoPE
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)

        # FlashAttention check
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension

        # 1. Project to Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Reshape to (Batch, Heads, Time, Head_Dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Optional RoPE positional encoding
        if self.use_rope:
            cos, sin = self.rope(v, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Scaled Dot-Product Attention
        # PyTorch MPS currently does not support dropout in scaled_dot_product_attention
        use_fast_attn = self.flash and (x.device.type != "mps" or not self.training)

        if use_fast_attn:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual mathematical formulation: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout > 0:
                att = self.attn_dropout(att)
            y = att @ v # (B, H, T, Head_Dim)

        # 5. Reassemble multi-head outputs: (B, H, T, Head_Dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Final linear projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

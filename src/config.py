"""
Model and Training Configuration for Large Language Model from Scratch.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Hyperparameter configuration for Decoder-Only Transformer LLM."""

    # Vocabulary & Context
    vocab_size: int = 65            # Number of unique tokens in vocabulary
    block_size: int = 256           # Maximum context window length (T)

    # Architecture dimensions
    n_layer: int = 6                # Number of Transformer blocks
    n_head: int = 6                 # Number of attention heads
    n_embd: int = 384               # Embedding dimension (d_model), must be divisible by n_head
    dropout: float = 0.2            # Dropout probability for regularization
    bias: bool = False              # Whether to include bias terms in Linear & LayerNorm layers

    # Positional Embedding Strategy
    use_rope: bool = False          # True for Rotary Position Embedding (RoPE), False for Learned Positional Embeddings
    norm_type: str = "layernorm"    # 'layernorm' or 'rmsnorm'
    tie_weights: bool = True        # Tie input token embeddings with output LM head projection

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"Embedding dimension n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
        )
        assert self.norm_type in ("layernorm", "rmsnorm"), (
            f"Unsupported norm_type: {self.norm_type}. Choose 'layernorm' or 'rmsnorm'."
        )

    @classmethod
    def gpt_nano(cls, vocab_size: int = 65, block_size: int = 128) -> "LLMConfig":
        """Ultra-lightweight configuration (~3.5M parameters), trains in 2-3 mins on CPU/Mac MPS."""
        return cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.1,
            bias=False,
            use_rope=False
        )

    @classmethod
    def gpt_micro(cls, vocab_size: int = 65, block_size: int = 256) -> "LLMConfig":
        """Balanced configuration (~10.8M parameters) for desktop pretraining with rich representation."""
        return cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.2,
            bias=False,
            use_rope=False
        )

    @classmethod
    def gpt2_small(cls, vocab_size: int = 50257, block_size: int = 1024) -> "LLMConfig":
        """Standard GPT-2 (124M parameter) architecture configuration."""
        return cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.1,
            bias=True,
            use_rope=False
        )

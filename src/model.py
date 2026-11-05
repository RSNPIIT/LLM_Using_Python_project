"""
Complete Decoder-Only Generative Pretrained Transformer Model.
Implements the full architecture from token embeddings to autoregressive sampling.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import LLMConfig
from .transformer import TransformerBlock, RMSNorm


class LargeLanguageModel(nn.Module):
    """
    A Decoder-Only Generative Pretrained Transformer (GPT-style LLM).
    Processes sequences autoregressively to predict the next token probability distribution.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd) if not config.use_rope else None,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd) if config.norm_type == "rmsnorm" else nn.LayerNorm(config.n_embd, elementwise_affine=True)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share parameters between input embeddings and output projection
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights with standard GPT-2 scheme
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with standard normal distribution (std=0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return total number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the LLM.
        idx: LongTensor of shape (Batch, Time) containing token IDs.
        targets: Optional LongTensor of shape (Batch, Time) containing next-token target IDs.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block_size is only {self.config.block_size}"
        )

        # 1. Token Embeddings: (B, T) -> (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)

        # 2. Positional Embeddings: (T, n_embd) -> broadcast to (B, T, n_embd)
        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        # 3. Stack of Transformer Blocks
        for block in self.transformer.h:
            x = block(x)

        # 4. Final Layer Normalization
        x = self.transformer.ln_f(x)

        # 5. Language Model Head Projection to Vocabulary
        if targets is not None:
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: compute logits only for the final position
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens conditioned on prompt token sequence idx.
        Supports:
        - Temperature scaling (higher = more creative/random, lower = more deterministic)
        - Top-K filtering (restricts sampling to top K highest probability tokens)
        - Top-P (Nucleus) filtering (restricts sampling to cumulative mass threshold P)
        """
        for _ in range(max_new_tokens):
            # Crop context if sequence exceeds maximum block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get logits for the last token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (Batch, vocab_size)

            # 1. Apply temperature scaling
            if temperature == 0.0:
                # Greedy argmax decoding
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                continue
            else:
                logits = logits / max(temperature, 1e-5)

            # 2. Optional Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # 3. Optional Top-P (Nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # 4. Convert logits to probabilities and sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 5. Append sampled token to running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights and configuration dict to file."""
        import dataclasses
        torch.save({
            "config": dataclasses.asdict(self.config),
            "state_dict": self.state_dict()
        }, filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str, map_location: str = "cpu") -> "LargeLanguageModel":
        """Load model from saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        config_data = checkpoint["config"]
        if isinstance(config_data, dict):
            config = LLMConfig(**config_data)
        else:
            config = config_data
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

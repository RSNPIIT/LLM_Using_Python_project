"""
Dataset and DataLoader utilities for autoregressive language modeling.
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    A PyTorch Dataset that yields sliding-window chunks of token sequences for causal language modeling.
    For an input sequence `x`, the target `y` is `x` shifted forward by exactly 1 position.
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int):
        assert len(token_ids) > block_size, (
            f"Dataset length ({len(token_ids)}) must be strictly greater than block_size ({block_size})."
        )
        self.data = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Chunk of size block_size
        x = self.data[idx : idx + self.block_size]
        # Next token targets for causal prediction
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_dataloaders(
    token_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    train_split: float = 0.9,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits token sequence into training and validation sets, returning ready-to-use DataLoaders.
    """
    n = int(train_split * len(token_ids))
    train_data = token_ids[:n]
    val_data = token_ids[n:]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=True
    )

    return train_loader, val_loader

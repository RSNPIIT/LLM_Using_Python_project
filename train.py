"""
Pretraining Harness for Large Language Model from Scratch.
Implements modern training loop with AdamW, Cosine LR Schedule, Perplexity Logging,
and live generation previews on CPU, Apple Silicon (MPS), or NVIDIA GPU (CUDA).
"""

import os
import sys
import time
import math
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.config import LLMConfig
from src.model import LargeLanguageModel
from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataloaders
from data.download_data import ensure_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Decoder-Only LLM from Scratch in Python")
    parser.add_argument("--data_path", type=str, default="data/input.txt", help="Path to text corpus")
    parser.add_argument("--preset", type=str, choices=["nano", "micro", "custom"], default="nano", help="Model preset")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context window length (T)")
    parser.add_argument("--n_layer", type=int, default=4, help="Transformer layers (if custom)")
    parser.add_argument("--n_head", type=int, default=4, help="Attention heads (if custom)")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension (if custom)")
    parser.add_argument("--max_iters", type=int, default=1000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate every N iterations")
    parser.add_argument("--eval_iters", type=int, default=20, help="Iterations to average for validation")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Linear warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="AdamW weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, mps, cuda, cpu)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(it: int, max_iters: int, warmup_iters: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(model: LargeLanguageModel, weight_decay: float, learning_rate: float, device_type: str):
    """
    Decoupled weight decay: applies weight decay to 2D matrices (embeddings and weights),
    while excluding 1D parameters (biases and layernorm/RMSnorm scales).
    Gracefully handles weight-tying between embedding and LM head.
    """
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"[*] Optimizer: {len(decay_params)} decay tensors ({num_decay_params:,} params), "
          f"{len(nodecay_params)} non-decay tensors ({num_nodecay_params:,} params)")

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer


@torch.no_grad()
def estimate_loss(model: LargeLanguageModel, data_loader, eval_iters: int, device: torch.device) -> float:
    """Estimates average loss across eval_iters batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    data_iter = iter(data_loader)
    for k in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = get_device(args.device)
    print("=" * 70)
    print("  🚀 LLM FROM SCRATCH IN PYTHON - PRETRAINING PIPELINE")
    print(f"  • Author: Ramrup Satpati (RSNPIIT)")
    print(f"  • Hardware Accelerator: {device.type.upper()}")
    print("=" * 70)

    # 1. Dataset & Tokenizer
    if not os.path.exists(args.data_path):
        ensure_dataset()
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharacterTokenizer.from_text(text)
    tokenizer_path = os.path.join(args.save_dir, "vocab.json")
    tokenizer.save(tokenizer_path)
    print(f"[*] Corpus loaded: {len(text):,} characters | Unique Vocabulary: {tokenizer.vocab_size} tokens")

    # Encode full dataset to tensor
    data_tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_loader, val_loader = create_dataloaders(
        data_tensor,
        block_size=args.block_size,
        batch_size=args.batch_size,
        train_split=0.9,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model Configuration
    if args.preset == "nano":
        config = LLMConfig.gpt_nano(vocab_size=tokenizer.vocab_size, block_size=args.block_size)
    elif args.preset == "micro":
        config = LLMConfig.gpt_micro(vocab_size=tokenizer.vocab_size, block_size=args.block_size)
    else:
        config = LLMConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd
        )

    model = LargeLanguageModel(config).to(device)
    print(f"[*] Architecture: {config.n_layer} Layers | {config.n_head} Heads | {config.n_embd} Embedding Dim")
    print(f"[*] Total Trainable Parameters: {model.get_num_params():,}")

    # 3. Optimizer & Scheduler
    optimizer = configure_optimizers(model, args.weight_decay, args.lr, device.type)

    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    start_time = time.time()

    print("\n[*] Starting Pretraining Loop...")
    print(f"{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val PPL':>9} | {'LR':>9} | {'Speed':>9}")
    print("-" * 65)

    for step in range(args.max_iters + 1):
        # Cosine learning rate update
        lr = get_lr(step, args.max_iters, args.warmup_iters, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic Evaluation & Sample Generation
        if step % args.eval_interval == 0 or step == args.max_iters:
            train_loss = estimate_loss(model, train_loader, args.eval_iters, device)
            val_loss = estimate_loss(model, val_loader, args.eval_iters, device)
            val_ppl = math.exp(min(val_loss, 20)) # Perplexity
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / max(elapsed, 1e-4)

            print(f"{step:6d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_ppl:9.2f} | {lr:9.2e} | {steps_per_sec:7.1f} it/s")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = os.path.join(args.save_dir, "best_model.pt")
                model.save_checkpoint(best_checkpoint)

            # Live text generation preview
            prompt_text = "\nKING:"
            prompt_tokens = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
            gen_tokens = model.generate(prompt_tokens, max_new_tokens=60, temperature=0.7, top_k=20)
            sample_str = tokenizer.decode(gen_tokens[0].tolist()).replace("\n", " ")
            print(f"   ↳ [Sample @ Step {step}]: \"{sample_str[:100]}...\"\n")

        if step == args.max_iters:
            break

        # Batch forward & backward step
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"  🎉 Training Complete in {total_time:.2f} seconds!")
    print(f"  • Best Validation Loss: {best_val_loss:.4f} (Perplexity: {math.exp(best_val_loss):.2f})")
    print(f"  • Checkpoint saved at: {os.path.join(args.save_dir, 'best_model.pt')}")
    print(f"  • Vocab saved at: {tokenizer_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

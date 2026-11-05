"""
Standalone Inference and Text Generation Script.
Samples tokens autoregressively from a trained LLM checkpoint.
"""

import os
import argparse
import torch

from src.model import LargeLanguageModel
from src.tokenizer import CharacterTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained LLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint file")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json", help="Path to vocabulary JSON file")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Initial text prompt")
    parser.add_argument("--max_tokens", type=int, default=300, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature (>0)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling limit")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (Nucleus) cumulative mass threshold")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, mps, cuda, cpu)")
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[!] Error: Checkpoint file '{args.checkpoint}' not found. Please train the model first using train.py.")
        return
    if not os.path.exists(args.vocab):
        print(f"[!] Error: Vocabulary file '{args.vocab}' not found.")
        return

    device = get_device(args.device)
    tokenizer = CharacterTokenizer.load(args.vocab)
    model = LargeLanguageModel.load_checkpoint(args.checkpoint, map_location=str(device))
    model.to(device)
    model.eval()

    print("=" * 65)
    print("  ✨ LLM GENERATION PIPELINE")
    print(f"  • Prompt: {repr(args.prompt)}")
    print(f"  • Temperature: {args.temperature} | Top-K: {args.top_k} | Top-P: {args.top_p}")
    print(f"  • Accelerator: {device.type.upper()}")
    print("=" * 65)

    # Encode prompt
    encoded_prompt = tokenizer.encode(args.prompt)
    if not encoded_prompt:
        # Fallback to space if prompt has unknown chars
        encoded_prompt = [0]
    prompt_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    # Autoregressive generation
    output_tokens = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    generated_text = tokenizer.decode(output_tokens[0].tolist())
    print("\n--- GENERATED COMPLETION ---")
    print(generated_text)
    print("----------------------------\n")


if __name__ == "__main__":
    main()

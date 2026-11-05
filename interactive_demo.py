"""
Interactive Terminal REPL for LLM Completion.
Allows dynamic prompting with live parameter tuning (temperature, top-k, top-p).
"""

import os
import sys
import torch

from src.model import LargeLanguageModel
from src.tokenizer import CharacterTokenizer


def main():
    checkpoint_path = "checkpoints/best_model.pt"
    vocab_path = "checkpoints/vocab.json"

    if not os.path.exists(checkpoint_path):
        print("[!] Error: No trained model found in checkpoints/best_model.pt")
        print("    Please run: python train.py --preset nano --max_iters 500")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = CharacterTokenizer.load(vocab_path)
    model = LargeLanguageModel.load_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    model.eval()

    temperature = 0.75
    top_k = 40
    top_p = 0.90
    max_tokens = 200

    print("=" * 65)
    print("  💬 Welcome to the Interactive LLM Shell!")
    print(f"  • Device: {device.type.upper()}")
    print("  • Commands:")
    print("      :temp <val>    - Set sampling temperature (e.g. :temp 0.8)")
    print("      :topk <val>    - Set Top-K filtering (e.g. :topk 50)")
    print("      :topp <val>    - Set Top-P nucleus threshold (e.g. :topp 0.95)")
    print("      :tokens <val>  - Set maximum tokens to generate")
    print("      :quit or :q    - Exit the shell")
    print("=" * 65)

    while True:
        try:
            prompt = input(f"\n[temp={temperature:.2f}, k={top_k}] User ❯ ")
            prompt = prompt.strip()
            if not prompt:
                continue
            if prompt in (":quit", ":q", "exit"):
                print("Goodbye!")
                break
            if prompt.startswith(":temp"):
                temperature = float(prompt.split()[1])
                print(f"[*] Temperature set to {temperature}")
                continue
            if prompt.startswith(":topk"):
                top_k = int(prompt.split()[1])
                print(f"[*] Top-K set to {top_k}")
                continue
            if prompt.startswith(":topp"):
                top_p = float(prompt.split()[1])
                print(f"[*] Top-P set to {top_p}")
                continue
            if prompt.startswith(":tokens"):
                max_tokens = int(prompt.split()[1])
                print(f"[*] Max tokens set to {max_tokens}")
                continue

            encoded = tokenizer.encode(prompt)
            if not encoded:
                encoded = [0]
            inp = torch.tensor([encoded], dtype=torch.long, device=device)

            print("\n🤖 LLM:", end=" ", flush=True)
            out = model.generate(inp, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            gen_text = tokenizer.decode(out[0].tolist())
            # Print only generated portion
            print(gen_text[len(prompt):].strip())

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

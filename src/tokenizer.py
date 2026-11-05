"""
Tokenization Engines for LLM from Scratch.
Includes Character-level Tokenizer (zero dependencies, completely transparent)
and Byte-Pair Encoding (BPE) subword tokenizer via tiktoken.
"""

import json
from typing import List, Dict, Optional


class CharacterTokenizer:
    """
    A pure-Python character-level tokenizer.
    Constructs a deterministic mapping between unique characters and integer token IDs.
    """

    def __init__(self, vocab_chars: Optional[List[str]] = None):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        if vocab_chars:
            self.build_vocab(vocab_chars)

    def build_vocab(self, chars: List[str]) -> None:
        """Build bidirectional lookup tables from a list or set of characters."""
        unique_sorted = sorted(list(set(chars)))
        self.char_to_id = {ch: i for i, ch in enumerate(unique_sorted)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_sorted)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        """Fit vocabulary directly on a text corpus."""
        unique_chars = sorted(list(set(text)))
        tokenizer = cls()
        tokenizer.build_vocab(unique_chars)
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token IDs."""
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of integer token IDs back into a string."""
        return "".join([self.id_to_char.get(i, "") for i in token_ids])

    def save(self, filepath: str) -> None:
        """Save vocabulary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "char_to_id": self.char_to_id,
                "vocab_size": self.vocab_size
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharacterTokenizer":
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.char_to_id = data["char_to_id"]
        tokenizer.id_to_char = {int(v): k for k, v in data["char_to_id"].items()}
        return tokenizer


class BPETokenizer:
    """
    Subword tokenizer based on Byte-Pair Encoding (BPE),
    wrapping OpenAI's tiktoken for standard GPT-2 / GPT-4 vocabulary.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids)

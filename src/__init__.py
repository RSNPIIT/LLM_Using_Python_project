"""
LLM From Scratch in Python
A clean, modular, and educational implementation of a Decoder-Only Generative Pretrained Transformer.
Author: Ramrup Satpati (RSNPIIT)
License: GNU General Public License v3.0 (GPLv3)
"""

from .config import LLMConfig
from .model import LargeLanguageModel
from .tokenizer import CharacterTokenizer, BPETokenizer

__version__ = "0.1.0"
__all__ = ["LLMConfig", "LargeLanguageModel", "CharacterTokenizer", "BPETokenizer"]

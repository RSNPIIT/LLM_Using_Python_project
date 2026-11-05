"""
Dataset Downloader and Fallback Generator for LLM Pretraining.
Downloads Tiny Shakespeare (1.1MB) or prepares local text data.
"""

import os
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "input.txt")
SAMPLE_FILE = os.path.join(DATA_DIR, "sample.txt")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


FALLBACK_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth.
"""


def ensure_dataset() -> str:
    """Ensures input.txt or sample.txt exists in data directory."""
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 1000:
        print(f"[*] Found existing training dataset at {DATA_FILE}")
        return DATA_FILE

    # Ensure sample.txt always exists
    if not os.path.exists(SAMPLE_FILE):
        with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
            f.write(FALLBACK_TEXT.strip() * 10)
        print(f"[*] Created local sample corpus at {SAMPLE_FILE}")

    print("[*] Downloading Tiny Shakespeare dataset...")
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_FILE)
        print(f"[*] Successfully downloaded dataset ({os.path.getsize(DATA_FILE) / 1024:.1f} KB) to {DATA_FILE}")
        return DATA_FILE
    except Exception as e:
        print(f"[!] Warning: Could not download remote dataset ({e}). Using local sample.txt fallback.")
        return SAMPLE_FILE


if __name__ == "__main__":
    path = ensure_dataset()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset preview: Length={len(text)} characters, Unique characters={len(set(text))}")

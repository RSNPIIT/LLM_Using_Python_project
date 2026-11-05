# Large Language Model (LLM) from Scratch in Python

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Hardware: CPU / MPS / CUDA](https://img.shields.io/badge/Hardware-CPU%20%7C%20Apple%20Silicon%20MPS%20%7C%20NVIDIA%20CUDA-purple.svg)](https://pytorch.org/)

> **A clean, modular, and educational implementation of a Decoder-Only Generative Pretrained Transformer (GPT) built from mathematical first principles in pure Python and PyTorch.**

Maintained and developed by **Ramrup Satpati ([RSNPIIT](https://github.com/RSNPIIT))**. Continuously open-source under the **GNU General Public License v3.0 (GPLv3)**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture Blueprint](#-architecture-blueprint)
- [Mathematical Foundations](#-mathematical-foundations)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
  - [1. Installation](#1-installation)
  - [2. Dataset Preparation](#2-dataset-preparation)
  - [3. Pretraining the LLM](#3-pretraining-the-llm)
  - [4. Generating Text](#4-generating-text)
  - [5. Interactive Terminal Shell](#5-interactive-terminal-shell)
- [Model Configurations](#-model-configurations)
- [Sampling Strategies](#-sampling-strategies)
- [License](#-license)

---

## 🌟 Overview

Most deep learning libraries abstract away what makes Large Language Models actually work. This repository provides an end-to-end, fully transparent implementation that guides you through:
1. **Tokenization**: Building a bidirectional character-level vocabulary and subword BPE tokenizers.
2. **Causal Attention**: Why masking future tokens prevents look-ahead bias and enforces autoregression.
3. **Transformer Blocks**: Pre-LayerNorm / RMSNorm, Multi-Head Self-Attention, GELU Feed-Forward Networks, and Residual connections.
4. **Pretraining Loop**: AdamW with decoupled weight decay, Cosine Annealing with linear warmup, and Perplexity metrics.
5. **Autoregressive Generation**: Nucleus (Top-p) sampling, Top-k filtering, and temperature scaling.

---

## 🏛️ Architecture Blueprint

```
                      [ Raw Text Corpus ]
                               │
                               ▼
                    [ Tokenizer Engine ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [ Input Tokens x: (B, T) ]            [ Target Tokens y: (B, T) ]
            │                                     │
            ▼                                     │
┌──────────────────────────────┐                  │
│ Token Embedding (wte)        │                  │
│ + Positional Embedding (wpe) │                  │
└──────────────┬───────────────┘                  │
               ▼                                  │
    [ Layer 1 .. Layer N ]                        │
 ┌────────────────────────────┐                   │
 │   Pre-LN / RMSNorm         │                   │
 │   Causal Multi-Head Attn   │                   │
 │   Residual Add (+)         │                   │
 │   Pre-LN / RMSNorm         │                   │
 │   FeedForward MLP (GELU)   │                   │
 │   Residual Add (+)         │                   │
 └─────────────┬──────────────┘                   │
               ▼                                  │
      [ Final LayerNorm ]                         │
               ▼                                  │
      [ LM Head Projection ] ──► Logits: (B, T, V)│
                                      │           │
                                      ▼           ▼
                         [ Cross-Entropy Loss L(θ) ]
```

---

## 📐 Mathematical Foundations

### 1. Scaled Dot-Product Attention
Given Query ($Q$), Key ($K$), and Value ($V$) representations with dimension $d_k$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where $M$ is the upper-triangular **causal attention mask**:

$$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

### 2. Multi-Head Projection
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 3. Objective Function & Perplexity
The model is trained on autoregressive next-token prediction via Cross-Entropy Loss:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_1, \dots, x_{t-1}; \theta)$$

$$\text{Perplexity} = \exp(\mathcal{L})$$

---

## 📂 Repository Structure

```
LLM_Using_Python_project/
├── LICENSE                 # GNU General Public License v3.0
├── README.md               # Architecture documentation & guide
├── requirements.txt        # Minimal dependencies
├── train.py                # Pretraining harness with live generation
├── generate.py             # Inference script with temperature & nucleus sampling
├── interactive_demo.py     # Real-time terminal chat / completion REPL
├── data/
│   ├── download_data.py    # Downloads Tiny Shakespeare benchmark corpus
│   └── sample.txt          # Fallback sample corpus
└── src/
    ├── __init__.py         # Package exports
    ├── config.py           # Hyperparameter configs (Nano, Micro, GPT-2)
    ├── tokenizer.py        # Character and BPE tokenizers
    ├── dataset.py          # Autoregressive sliding-window DataLoader
    ├── attention.py        # Causal Multi-Head Self-Attention & RoPE
    ├── transformer.py      # RMSNorm, LayerNorm, FeedForward & Blocks
    └── model.py            # Complete LargeLanguageModel module
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone your repository and activate your environment:

```bash
git clone git@github.com:RSNPIIT/LLM_Using_Python_project.git
cd LLM_Using_Python_project
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the canonical **Tiny Shakespeare** (1.1 MB) dataset with one command:

```bash
python data/download_data.py
```

### 3. Pretraining the LLM

Train the **Nano** architecture on your machine (automatically detects Apple Silicon MPS, NVIDIA CUDA, or CPU):

```bash
python train.py --preset nano --max_iters 1000 --batch_size 32 --eval_interval 100
```

Watch the model learn in real time as validation loss and perplexity drop:

```
  Step | Train Loss |   Val Loss |   Val PPL |        LR |     Speed
-----------------------------------------------------------------
     0 |     4.2022 |     4.2099 |     67.35 |  1.20e-05 |     0.8 it/s
   100 |     2.4180 |     2.4312 |     11.37 |  5.80e-04 |    24.2 it/s
   500 |     1.6541 |     1.7820 |      5.94 |  4.10e-04 |    24.5 it/s
  1000 |     1.3812 |     1.5420 |      4.67 |  6.00e-05 |    24.6 it/s
```

Checkpoints and vocabulary are saved automatically to `checkpoints/best_model.pt` and `checkpoints/vocab.json`.

### 4. Generating Text

Generate completions conditioned on your own custom prompt:

```bash
python generate.py \
    --prompt "To be, or not to be, that is the " \
    --max_tokens 250 \
    --temperature 0.75 \
    --top_k 40 \
    --top_p 0.90
```

### 5. Interactive Terminal Shell

Launch an interactive prompt session to experiment with temperatures and sampling parameters on the fly:

```bash
python interactive_demo.py
```

```text
  💬 Welcome to the Interactive LLM Shell!
  • Device: MPS
  • Commands: :temp <float>, :topk <int>, :topp <float>, :tokens <int>, :quit

[temp=0.75, k=40] User ❯ KING:
🤖 LLM: What news from the northern march, my lord?
```

---

## ⚙️ Model Configurations

| Preset | Layers | Heads | Embedding Dim | Context ($T$) | Approx Params | Best Suited For |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`gpt_nano`** | 4 | 4 | 128 | 128 | ~0.8M | Fast local testing, laptops, CPU |
| **`gpt_micro`** | 6 | 6 | 384 | 256 | ~10.8M | High quality desktop training on MPS/GPU |
| **`gpt2_small`**| 12 | 12 | 768 | 1024 | ~124M | Full GPT-2 architecture benchmark |

---

## 🎯 Sampling Strategies

- **Greedy Search (`temp = 0.0`)**: Selects the token with maximum probability $\arg\max P(x_t)$. Fast and deterministic.
- **Temperature Scaling ($T > 0$)**: Rescales logits $z_i / T$ before softmax. Lower temperature produces focused, conservative text; higher temperature encourages creative, varied vocabulary.
- **Top-K Filtering**: Keeps only the top $K$ most likely tokens, cutting off the long tail of low-probability noise.
- **Top-P (Nucleus) Sampling**: Dynamically selects the smallest set of tokens whose cumulative probability exceeds threshold $P$ (e.g. 0.90), adapting context-by-context.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the [LICENSE](LICENSE) file for complete details.

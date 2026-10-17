#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 23:13:29 2026

@author: sivakumarannadurai
"""

# Build a GPT from Scratch in PyTorch

This repository contains a complete, step-by-step implementation of a miniature Generative Pre-trained Transformer (GPT) language model built entirely from scratch using **PyTorch**, inspired by Andrej Karpathy's famous nanoGPT. 

The model is trained on the Tiny Shakespeare dataset to generate Shakespeare-like character text.

---

## Features
* **Custom Tokenizer:** Character-level encoding and decoding mappings.
* **Self-Attention Mechanism:** Multi-head self-attention implementation from scratch.
* **Transformer Blocks:** Residual connections, layer normalization, and feed-forward networks.
* **Training Pipeline:** Batch generation, loss computation using cross-entropy, and AdamW optimization.
* **Model Checkpointing:** Code included to save and load trained model weights (`.pt`).

---

## How to Run
You can run this code directly in your browser using **Google Colab**:

1. Open a new notebook in [Google Colab](https://colab.research.google.com/).
2. Copy and paste the script from `model.py` into your notebook cells.
3. Make sure to enable a **GPU runtime** (`Runtime -> Change runtime type -> T4 GPU`).
4. Run the cells sequentially to train the model and generate text!

---

## Requirements
* Python 3.8+
* PyTorch
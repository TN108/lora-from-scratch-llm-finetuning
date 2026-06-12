# LoRA Fine-Tuning Assignment

This repository contains an implementation of Low-Rank Adaptation (LoRA) in PyTorch. The project first demonstrates LoRA on a simple linear layer and then applies LoRA to selected attention projection layers of a pre-trained causal language model.

## Project Overview

Large language models contain millions or billions of parameters. Full fine-tuning requires updating all model weights, which is expensive in terms of memory and computation.

LoRA solves this by freezing the original model weights and training only small low-rank matrices. Instead of learning a full weight update:

\[
W_{\text{updated}} = W + \Delta W
\]

LoRA approximates the update as:

\[
\Delta W \approx BA
\]

where `A` and `B` are small trainable matrices.

## Main Components

This project includes:

- A standalone `LoRALayer`
- A non-merged `LoRALinear` wrapper
- A merged LoRA linear layer for inference
- Mathematical equivalence verification between merged and non-merged LoRA
- Dataset preparation using AG News
- LoRA application to GPT-Neo-125M attention layers
- Freezing of base model parameters
- Trainable parameter inspection
- Parameter count comparison
- LoRA training loop
- Token-level accuracy evaluation

## Model Used

The language model used in this assignment is:

```text
EleutherAI/gpt-neo-125M

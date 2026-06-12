# LoRA Fine-Tuning Assignment

## Overview

This project implements Low-Rank Adaptation (LoRA) from scratch in PyTorch and applies it to a pre-trained causal language model. The assignment begins with a simple LoRA implementation for a linear layer, then extends the same idea to attention projection layers inside `EleutherAI/gpt-neo-125M`.

LoRA is a parameter-efficient fine-tuning method. Instead of updating all original model weights, the base model is frozen and only small low-rank matrices are trained. This reduces memory usage and training cost while preserving the original pre-trained model parameters.

## Objectives

The main objectives of this assignment are:

- Implement a standalone LoRA layer using two trainable low-rank matrices.
- Wrap an existing `nn.Linear` layer with LoRA adaptation.
- Implement both non-merged and merged LoRA versions.
- Verify mathematical equivalence between merged and non-merged LoRA.
- Prepare a text dataset for causal language modeling.
- Apply LoRA to attention projection layers in GPT-Neo-125M.
- Freeze base model parameters and train only LoRA parameters.
- Compare parameter counts between full fine-tuning and LoRA fine-tuning.
- Train and evaluate the LoRA-adapted language model.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Google Colab
- AG News Dataset
- GPT-Neo-125M

## Model and Dataset

The pre-trained model used in this assignment is:

```text
EleutherAI/gpt-neo-125M
```

The dataset used is:

```text
AG News
```

A subset of the AG News dataset is loaded, shuffled, tokenized, and formatted for causal language modeling. Labels are constructed from the input token IDs, and padding positions are masked using `-100` so that they do not contribute to the training loss.

## LoRA Method

In full fine-tuning, a model learns a full weight update:

```text
W_updated = W + ΔW
```

In LoRA, the update matrix is approximated using two smaller matrices:

```text
ΔW ≈ B A
```

The final adapted weight is:

```text
W_updated = W + (alpha / rank) * B A
```

where:

- `rank` controls the low-rank dimension.
- `alpha` controls the scaling strength.
- `A` and `B` are trainable low-rank matrices.
- The original weight matrix `W` remains frozen.

## Implemented Components

### 1. LoRALayer

The `LoRALayer` class computes the low-rank update independently from the base layer. It uses two trainable linear layers:

```text
lora_A: input_features -> rank
lora_B: rank -> output_features
```

The output is computed as:

```text
LoRA(x) = lora_B(lora_A(x)) * (alpha / rank)
```

The `lora_B` matrix is initialized to zeros, so the LoRA update is initially zero. This means the adapted model initially behaves exactly like the original model.

### 2. LoRALinear

The `LoRALinear` class wraps an existing `nn.Linear` layer. The original linear layer is frozen, and the LoRA update is added during the forward pass.

The non-merged output is:

```text
output = frozen_linear(x) + lora(x)
```

This version is suitable for training because the base parameters and LoRA parameters remain separate.

### 3. LinearWithLoRAMerged

The merged LoRA implementation computes the LoRA weight update and adds it directly to the original weight matrix:

```text
merged_weight = W + ΔW
```

The linear operation is then performed using the merged weight. This version is useful for inference because it avoids computing the base layer and LoRA layer separately.

### 4. Mathematical Equivalence Check

The merged and non-merged versions were initialized with the same base weights and LoRA matrices. Their outputs were compared using the maximum absolute difference.

Result:

```text
Maximum absolute difference: 0.0
```

This confirms that the merged and non-merged LoRA implementations are mathematically equivalent.

## Applying LoRA to GPT-Neo-125M

LoRA was applied to the attention projection layers of GPT-Neo-125M. The selected target modules were:

```text
q_proj
k_proj
v_proj
out_proj
```

These correspond to the query, key, value, and output projections in the self-attention mechanism.

All original model parameters were frozen. Only the LoRA matrices were enabled for training.

## Trainable Parameter Verification

After applying LoRA, only the following parameter types were trainable:

```text
lora_A.weight
lora_B.weight
```

This confirmed that the base model parameters were successfully frozen and only the LoRA adapters were trainable.

## Parameter Count Results

The base model parameter count was:

```text
Total parameters: 125,198,592
Trainable parameters: 0
```

The LoRA-adapted model parameter count was:

```text
Total parameters: 125,788,416
Trainable parameters: 589,824
```

For full fine-tuning, all base model parameters would be trainable:

```text
Full fine-tuning trainable parameters: 125,198,592
```

With LoRA, only the following number of parameters were trained:

```text
LoRA trainable parameters: 589,824
```

The percentage of trainable parameters used by LoRA was:

```text
0.4711%
```

The reduction in trainable parameters was:

```text
99.5289%
```

This demonstrates that LoRA provides a highly parameter-efficient fine-tuning approach.

## Training

The LoRA-adapted model was trained using the prepared AG News dataset. The optimizer was configured to update only parameters where `requires_grad=True`, which means only LoRA matrices were updated.

The training loop included:

- Moving each batch to the selected device.
- Computing model loss using causal language modeling labels.
- Backpropagating the loss.
- Updating only LoRA parameters.
- Printing loss during training.

## Evaluation

Both the frozen base model and the LoRA-adapted model were evaluated on the test dataset using token-level accuracy.

For causal language modeling, logits and labels were shifted so that each token position predicts the next token. Padding positions were ignored using the `-100` label mask.

The evaluation compared:

```text
Base model token-level accuracy
LoRA model token-level accuracy
```

This comparison measures whether LoRA fine-tuning improved performance relative to the frozen pre-trained model.

## Key Results

- LoRA was successfully implemented from scratch.
- The original linear layer and language model parameters were frozen correctly.
- Only low-rank LoRA matrices were trainable.
- Merged and non-merged LoRA implementations produced identical outputs.
- LoRA reduced trainable parameters by approximately 99.53%.
- The LoRA-adapted model was trained efficiently using only a small fraction of the full model parameters.

## How to Run

Install the required libraries:

```bash
pip install torch transformers datasets
```

Run the notebook cells in the following order:

1. Import libraries.
2. Define `LoRALayer`.
3. Define `LoRALinear`.
4. Test LoRA on a simple linear layer.
5. Define merged LoRA implementation.
6. Verify equivalence between merged and non-merged LoRA.
7. Load tokenizer and GPT-Neo-125M.
8. Prepare the AG News dataset.
9. Apply LoRA to attention layers.
10. Freeze base parameters and enable LoRA parameters.
11. Count trainable parameters.
12. Train the LoRA-adapted model.
13. Evaluate base and LoRA models.

## AI Usage Statement

Generative AI was used as a support tool for understanding the assignment, structuring the implementation, debugging errors, and writing explanations. The generated code and explanations were reviewed, executed, tested, and modified before submission.

Tool used:

```text
ChatGPT 5.5 Thinking, OpenAI Plus Account, 2026
```

AI assistance was used for:

- Explaining LoRA concepts.
- Implementing LoRA layers.
- Debugging class definition and parameter-freezing issues.
- Preparing dataset, training, and evaluation code.
- Writing report and README content.

The final implementation was checked by running the notebook and verifying outputs such as trainable parameter counts and mathematical equivalence.

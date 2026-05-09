# Toxic Speech Detection and Explanation

A research-oriented NLP framework for toxic and implicit hate speech detection, target-group identification, and explanation generation using parameter-efficient fine-tuning techniques on encoder-decoder language models.

This project extends the ToXCL framework by exploring:
- Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
- Soft Target Group Conditioning using learned target-group embeddings
- Joint toxicity classification and explanation generation
- Multi-task learning with classification and language modeling objectives

The system is built using Flan-T5 and trained on implicit hate speech corpora for interpretable toxic speech analysis.

---

## Features

- Toxic and implicit hate speech classification
- Natural language explanation generation
- Target-group identification
- Parameter-efficient fine-tuning using LoRA
- Encoder-decoder architecture based on Flan-T5
- Multi-task training with classification and generation objectives
- Soft target-group conditioning using learned embeddings
- Conditional explanation generation
- HuggingFace Transformers + PEFT integration

---

# Architecture Overview

The framework extends the original ToXCL architecture with two primary improvements:

## 1. LoRA-Based Fine-Tuning

Instead of fully fine-tuning all Flan-T5 parameters, Low-Rank Adaptation (LoRA) was applied to the query and value projection matrices inside the encoder self-attention layers.

This significantly reduced:
- GPU memory usage
- Training cost
- Number of trainable parameters

while preserving downstream task performance.

LoRA was applied across all 12 encoder layers of Flan-T5.

---

## 2. Soft Target Group Conditioning

Rather than using target-group information only as text prompts, the system learns dedicated target-group embeddings and injects them directly into:
- the student classifier
- the explanation decoder

This allows the model to attend to target-group representations during generation and classification.

---

## Model Components

- Flan-T5 Encoder-Decoder Backbone
- Student Toxicity Classifier
- Explanation Decoder
- Target Group Generator
- Target Group Embeddings
- LoRA Attention Adapters
- Conditional Decoding Constraint
- Multi-Task Loss Optimization

---

## Datasets

The framework was trained and evaluated on:
- Implicit Hate Corpus (IHC)
- Toxic speech and hate speech benchmark datasets

The datasets contain:
- Toxic / non-toxic labels
- Target groups
- Human-written explanations

---

## LoRA Configuration

### Motivation

Full fine-tuning of Flan-T5-base (~248M parameters) is computationally expensive and prone to overfitting on smaller domain-specific datasets.

LoRA reduces trainable parameters by decomposing weight updates into low-rank matrices.

---

## LoRA Formula

Instead of directly updating:

```math
W
```

LoRA learns:

```math
ΔW = A × B
```

and modifies the forward pass as:

```math
y = Wx + (α/r)ABx
```

Only the low-rank matrices are trainable while the original model weights remain frozen.

---

## Parameter Efficiency

| Model | Total Parameters | Trainable Parameters |
|---|---|---|
| Full Flan-T5 Fine-Tuning | 248M | 248M |
| RoBERTa Teacher Classifier | 354M | 354M |
| Flan-T5 + LoRA | 248M + ~0.9M | ~0.9M |

---

## Soft Target Group Conditioning

Target groups are:
1. Extracted from posts
2. Embedded into learned vector representations
3. Pooled into target-group embeddings
4. Injected into both classification and generation modules

This creates stronger semantic conditioning compared to prompt-only target-group injection.

---

# Training Objectives

The framework jointly optimizes:

- Toxicity Classification Loss
- Language Modeling Loss
- Knowledge Distillation Loss (teacher-guided setup)

For the LoRA-only variant without teacher forcing:

```math
L = αL_cls + βL_clm
```

---

# Results

## LoRA Variant

| Metric | Score |
|---|---|
| Accuracy | 67.94 |
| METEOR | 67.94 |
| BERTScore | 67.94 |

The LoRA-based approach achieved strong parameter efficiency while requiring only ~0.9M trainable parameters. :contentReference[oaicite:0]{index=0}

---

## Soft Target Group Conditioning

| Metric | Score |
|---|---|
| Accuracy | 69.61 |
| METEOR | 67.89 |
| ROUGE-L | 67.94 |

Soft target-group conditioning improved classification performance by creating stronger semantic alignment between target groups and toxic intent. :contentReference[oaicite:1]{index=1}

---

# Example Outputs

The framework jointly predicts:
- Toxicity label
- Explanation
- Target groups

Example:

### Input Post
```text
part of your white heritage is being ruled by jews
```

### Predicted Label
```text
hate
```

### Predicted Explanation
```text
jews hate white people
```

### Predicted Target Groups
```text
Asian, Caucasian, Jewish
```

---

# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- PEFT (LoRA)
- Flan-T5
- NumPy
- Pandas

---

# Project Structure

```text
ToxicSpeechDetection/
│
├── toxcl.py
├── train.py
├── inference.py
├── datasets/
├── checkpoints/
├── outputs/
├── plots/
├── README.md
```

---

# Installation

```bash
pip install torch transformers peft datasets accelerate
```

---

# Usage

Train the model:

```bash
python train.py
```

Run inference:

```bash
python inference.py
```

---

# Research Focus

This work focuses on:
- Explainable hate speech detection
- Parameter-efficient adaptation of large language models
- Implicit toxicity understanding
- Multi-task NLP learning
- Controllable explanation generation

---

# References

- ToXCL: A Unified Framework for Toxic Speech Detection and Explanation
- LoRA: Low-Rank Adaptation of Large Language Models
- HateBERT
- Implicit Hate Corpus (IHC)
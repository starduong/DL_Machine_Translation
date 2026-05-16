# 🌐 English-Vietnamese Machine Translation

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37726?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

## ✨ Introduction

This is a **comprehensive educational project** for **Neural Machine Translation (NMT)** designed for beginners in NLP and Deep Learning. It demonstrates how to build, train, and evaluate two modern translation model architectures:

### 🎓 Learning Objectives

- Understand the full end-to-end process of building a machine translation model
- Learn how to implement theory into code for **Seq2Seq**, **Attention**, and **Transformer** architectures
- Get familiar with NLP and Deep Learning best practices
- Practice efficient handling of **large-scale data** (~2.88M sentence pairs)

---

## 🎯 Key Features

| #   | Feature                     | Details                                            |
| --- | --------------------------- | -------------------------------------------------- |
| ✅  | **Two Architectures**       | Seq2Seq + Attention vs. Transformer                |
| ✅  | **BPE Tokenizer**           | Professional subword preprocessing                 |
| ✅  | **EDA Analysis**            | Data exploration and anomaly detection             |
| ✅  | **Training Optimization**   | Label smoothing, learning rate schedulers          |
| ✅  | **Multi-Metric Evaluation** | BLEU, ROUGE, METEOR, inference                     |
| ✅  | **Visualization**           | Training curves, attention heatmaps, word clouds   |
| ✅  | **Pre-trained Checkpoints** | Best training checkpoints included                 |
| ✅  | **Jupyter Notebook**        | Clear, educational code with detailed explanations |

---

## 📊 Dataset

### Dataset Source

- **Dataset Name**: IWSLT mt_eng_vietnamese
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/ncduy/mt-en-vi)
- **Overall Size**:
  - Train: ~2,880,000 bilingual sentence pairs
  - Validation: ~11,000 pairs
  - Test: ~12,000 pairs
- **Languages**: English ↔ Vietnamese
- **Domain**: International Workshop on Spoken Language Translation (IWSLT)

### Example Data

```
EN: "Hello, how are you?"
VI: "Xin chào, bạn khỏe không?"

EN: "What is your name?"
VI: "Tên bạn là gì?"
```

### Data Processing

1. **Cleaning**: Remove empty lines and noisy characters
2. **Tokenization**: Use BPE (Byte Pair Encoding)
3. **Padding & Masking**: Prepare batches for training
4. **Vocabulary**: Build separate English and Vietnamese vocabularies

---

## 🏗️ Model Architectures

### 1️⃣ Seq2Seq + Attention Model (Bahdanau)

```
┌─────────────────────────────────────────────────────────────┐
│                  SEQ2SEQ + ATTENTION ARCHITECTURE            │
└─────────────────────────────────────────────────────────────┘

                        INPUT SEQUENCE
                            ↓
                        ┌───────────────────────┐
                        │  ENCODER (GRU)        │
                        │  (Bidirectional)      │
                        ├───────────────────────┤
                        │ • Input Embedding     │
                        │ • 2 GRU Layers        │
                        │ • Output: Context Vec │
                        └───────────────────────┘
                            ↓
                            ├─→ Initial Hidden State
                            └─→ Attention Context
                            ↓
                        ┌───────────────────────┐
                        │  ATTENTION MECHANISM  │
                        │  (Bahdanau Style)     │
                        ├───────────────────────┤
                        │ • Query = Decoder Out │
                        │ • Keys = Encoder Out  │
                        │ • Compute α (scores)  │
                        │ • Weighted Context    │
                        └───────────────────────┘
                            ↓
                        ┌───────────────────────┐
                        │  DECODER (GRU)        │
                        │  (Single Direction)   │
                        ├───────────────────────┤
                        │ • Input Embedding     │
                        │ • 2 GRU Layers        │
                        │ • Attention Input     │
                        │ • Output: Vocab Logits
                        └───────────────────────┘
                            ↓
                        OUTPUT SEQUENCE
```

**Model Details**:

- Encoder: 2-layer Bidirectional GRU (256 units)
- Decoder: 2-layer GRU (256 units)
- Attention: Bahdanau-style attention
- Embedding dimension: 256
- Vocabulary size: ~10,000 subwords (BPE)

### 2️⃣ Transformer Model

```
┌──────────────────────────────────────────────────────────────┐
│              TRANSFORMER ARCHITECTURE                        │
└──────────────────────────────────────────────────────────────┘

                INPUT SEQUENCE
                    ↓
                ┌────────────────────────┐
                │ INPUT EMBEDDING        │
                │ + POSITIONAL ENCODING  │
                └────────────────────────┘
                    ↓
                    ├─→ ┌────────────────────────────────┐
                    │   │  ENCODER STACK (N=6 layers)   │
                    │   ├────────────────────────────────┤
                    │   │ ┌──────────────────────────┐  │
                    │   │ │ MULTI-HEAD ATTENTION    │  │
                    │   │ │ (8 heads, 512 hidden)   │  │
                    │   │ └──────────────────────────┘  │
                    │   │ + Layer Norm                   │
                    │   │ ↓                              │
                    │   │ ┌──────────────────────────┐  │
                    │   │ │ FEED-FORWARD NETWORK    │  │
                    │   │ │ (2 Dense Layers)        │  │
                    │   │ └──────────────────────────┘  │
                    │   │ + Layer Norm                   │
                    │   └────────────────────────────────┘
                    ↓
                    ├─→ ┌────────────────────────────────┐
                    │   │ DECODER STACK (N=6 layers)    │
                    │   ├────────────────────────────────┤
                    │   │ ┌──────────────────────────┐  │
                    │   │ │ MASKED MULTI-HEAD ATTN   │  │
                    │   │ │ (Prevent looking ahead)  │  │
                    │   │ └──────────────────────────┘  │
                    │   │ + Layer Norm                   │
                    │   │ ↓                              │
                    │   │ ┌──────────────────────────┐  │
                    │   │ │ CROSS ATTENTION          │  │
                    │   │ │ (Attends to Encoder)     │  │
                    │   │ └──────────────────────────┘
                    │   │ + Layer Norm                   │
                    │   │ ↓                              │
                    │   │ ┌──────────────────────────┐  │
                    │   │ │ FEED-FORWARD NETWORK    │  │
                    │   │ │ (2 Dense Layers)        │  │
                    │   │ └──────────────────────────┘
                    │   │ + Layer Norm                   │
                    │   └────────────────────────────────┘
                    ↓
                ┌──────────────────────────┐
                │ OUTPUT LINEAR PROJECTION │
                │ + SOFTMAX OVER VOCAB     │
                └──────────────────────────┘
                    ↓
                OUTPUT SEQUENCE
```

**Model Details**:

- Model dimension: 512
- Number of layers: 6 encoder + 6 decoder
- Multi-head attention: 8 heads
- Feed-forward hidden size: 2048
- Positional encoding: sinusoidal
- Dropout: 0.1
- Maximum sequence length: 256

### Model Comparison

| Criterion            | Seq2Seq + Attention | Transformer        |
| -------------------- | ------------------- | ------------------ |
| **Training Speed**   | Slower (sequential) | Faster (parallel)  |
| **Memory Usage**     | Lower               | Higher             |
| **Long Dependency**  | Moderate            | Strong             |
| **Interpretability** | Easier              | More complex       |
| **Production Use**   | Less common         | Widely used (SOTA) |

---

## 🚀 Quick Start

### 🎯 Run the Notebook

```bash
# Open the Jupyter notebook
jupyter notebook en-vi-machine-translation.ipynb
```

### 📍 Main Steps

1. **Cells 1-2**: Install libraries and imports
2. **Cells 3-5**: Load dataset and perform EDA
3. **Cells 6-8**: Train the BPE tokenizer
4. **Cells 9-12**: Define the Seq2Seq model
5. **Cells 13-15**: Train Seq2Seq
6. **Cells 16-18**: Evaluate Seq2Seq
7. **Cells 19-22**: Define the Transformer model
8. **Cells 23-25**: Train Transformer
9. **Cells 26+**: Compare models and run inference

### 🎬 Run a Small Demo

If you run on CPU or have limited RAM, change:

```python
# In Cell 5 - Load data
MAX_SAMPLES  = 10_000   # instead of 100_000
MAX_VAL_TEST = 500      # instead of 5_000
```

---

## 📚 Detailed Documentation

### Part 1️⃣: Data Preparation

**Goal**: Understand the data before building the model

- ✅ Load the IWSLT dataset from Hugging Face
- ✅ Basic statistics (size, examples)
- ✅ Detect and clean noisy data
- ✅ Analyze sentence length distribution
- ✅ Extract common vocabulary statistics

**Output**: Clean dataset and EDA results

---

### Part 2️⃣: Text Preprocessing (Tokenization)

**Goal**: Convert raw text into tokens and embeddings

- ✅ **BPE Tokenizer**: Train on the training set
  - Handle rare words
  - Handle OOV (out-of-vocabulary) tokens
- ✅ **Vocabulary**: Build English and Vietnamese vocabularies
- ✅ **Encoding/Decoding**: Text ↔ ID sequences
- ✅ **Padding & Masking**: Prepare training batches

**Output**:

- `tokenizer_en.json` & `tokenizer_vi.json`
- Vocabulary: EN (~10K tokens) & VI (~10K tokens)

---

### Part 3️⃣: Build the Seq2Seq + Attention Model

**Goal**: Learn the encoder-decoder-attention mechanism

**Architecture**:

```
Encoder: Bidirectional GRU (2 layers)
  ↓ (Context Vector)
Attention: Bahdanau attention
  ↓
Decoder: Unidirectional GRU (2 layers)
  ↓
Output: Vocabulary softmax
```

**What you learn**:

- How GRU works
- What attention mechanisms do
- Teacher forcing in training
- Greedy decoding vs beam search

**Output**: `seq2seq_best.pth` (best checkpoint)

---

### Part 4️⃣: Build the Transformer Model

**Goal**: Understand the modern Transformer architecture (SOTA)

**Architecture**:

```
Positional Encoding
  ↓
Multi-Head Self-Attention (Scaled Dot-Product)
  ↓
Feed-Forward Network (FFN)
  ↓
(Repeat N times)
```

**What you learn**:

- Self-attention vs cross-attention
- Scaled dot-product attention
- Position-wise feed-forward networks
- Encoder-decoder interaction

**Output**: `transformer_best.pth` (best checkpoint)

---

### Part 5️⃣: Training & Optimization

**Techniques**:

- **Label smoothing**: avoid overfitting
- **Learning rate scheduling**: warm-up and decay
- **Gradient clipping**: prevent exploding gradients
- **Mixed precision** (optional): faster training

**Monitoring**:

- Training loss and validation loss
- BLEU score on the validation set
- Gradient magnitude and weight statistics

---

### Part 6️⃣: Model Evaluation

**Metrics**:

1. **BLEU Score** (0-100):
   - 0-20: Poor
   - 20-40: Fair
   - 40-60: Good
   - 60+: Excellent

2. **ROUGE Score** (Recall-oriented):
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence

3. **METEOR**: Semantic similarity

**Inference Modes**:

- **Greedy**: select argmax (fastest)
- **Beam Search** (k=5): choose top-k candidates (slower but better)
- **Nucleus Sampling**: probabilistic and diverse

---

## 🔄 Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│             DATA PROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

1. LOAD DATASET
   ├─ Source: Hugging Face IWSLT
   ├─ Splits: train (2.88M) | val (11K) | test (12K)
   └─ Format: (en_sentence, vi_sentence)

                    ↓

2. DATA CLEANING
   ├─ Remove empty lines
   ├─ Remove duplicates
   ├─ Remove special characters
   ├─ Normalize whitespace
   └─ Length filtering: 1-100 tokens

                    ↓

3. BPE TOKENIZATION
   ├─ Train BPE on training set
   │  └─ Vocabulary size: ~10K
   ├─ Build vocabulary
   │  ├─ Special tokens: [PAD], [UNK], [CLS], [SEP], [EOS]
   │  └─ Content tokens: ~10K subword units
   └─ Encode all splits

                    ↓

4. DATASET CREATION
   ├─ Create PyTorch Dataset class
   ├─ Implement collate_fn for batching
   ├─ Pad sequences to max length
   ├─ Create attention masks
   └─ Prepare DataLoader

                    ↓

5. BATCH PROCESSING (During Training)
   ├─ Load batch: (src_tokens, tgt_tokens)
   ├─ Create padding mask
   ├─ Move to GPU
   ├─ Forward pass → Loss
   └─ Backprop → Update weights

                    ↓

6. EVALUATION
   ├─ Generate predictions on validation set
   ├─ Compute BLEU/ROUGE scores
   ├─ Early stopping check
   └─ Save best checkpoint

                    ↓

7. INFERENCE
   ├─ Load trained model & tokenizer
   ├─ Tokenize input
   ├─ Beam search decoding
   ├─ Detokenize output
   └─ Return translation
```

**Referenced Papers**:

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) - Sutskever et al. (2014)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Bahdanau et al. (2015)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)

---

---

<div align="center">

### ⭐ If this project helps you, please give it a **star**! ⭐

</div>

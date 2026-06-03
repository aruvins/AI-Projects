# 🔵 Project 8 — Transformer From Scratch

## 🎯 Goal

Build a simplified Transformer architecture from scratch using PyTorch.

This project introduces the core ideas that power modern large language models such as GPT, BERT, and many state-of-the-art NLP systems.

Instead of using PyTorch's built-in Transformer modules, the model is implemented manually to develop an intuitive understanding of:

* token embeddings
* positional encoding
* self-attention
* transformer encoder blocks
* sequence classification

The model is trained on the IMDB movie review dataset to perform sentiment classification.

---

# 🏗 Project Structure

```text
project_8_transformer_from_scratch/

│
├── main.py
│
├── models/
│   ├── attention.py
│   ├── encoder.py
│   ├── positional_encoding.py
│   └── transformer.py
│
├── utils/
│   ├── dataset.py
│   ├── predictor.py
│   ├── trainer.py
│   └── visualize.py
│
├── outputs/
│   ├── transformer_model.pth
│   ├── training_loss.png
│   ├── attention_heatmap.png
│   ├── positional_encoding.png
│   └── sample_predictions.txt
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_8_transformer_from_scratch

```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate

```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Run the Project

```bash
python main.py
```

---

# 🧠 Concepts Learned

## Tokenization

Neural networks cannot process raw text directly.

Text is converted into tokens:

```text
"This movie was amazing"

↓

["this", "movie", "was", "amazing"]

↓

[17, 42, 8, 119]
```

---

## Word Embeddings

Each token is mapped to a learned vector representation.

Instead of representing words as simple IDs:

```text
movie = 42
```

the model learns:

```text
movie = [0.23, -1.12, 0.91, ...]
```

Words with similar meanings tend to occupy nearby regions of embedding space.

---

## Positional Encodings

Positional encoding is one of the most elegant ideas in the original Transformer paper because it solves a fundamental problem:

> Self-attention has no inherent notion of sequence order.

An RNN processes words one at a time:

```text
"The" → "movie" → "was" → "great"
```

so order is built into the computation.

A Transformer processes all words simultaneously:

```text
["The", "movie", "was", "great"]
```

which means the model initially sees a set of tokens, not a sequence.

Without positional information, these two sentences would look identical:

```text
Dog bites man
```

```text
Man bites dog
```

The same words exist in both sentences.

The Transformer needs a way to know where each word occurs.

---

# Step 1: Word Embeddings

Suppose we have:

```text
"The movie was great"
```

After tokenization:

```text
[17, 42, 8, 119]
```

The embedding layer converts each token into a vector.

Maybe:

```text
"The"    → [0.3, 0.7, -0.2, ...]
"movie"  → [1.2, 0.1,  0.5, ...]
"was"    → [-0.4, 0.9, 0.3, ...]
"great"  → [2.1, 1.8, 0.7, ...]
```

Stacked together:

```text
[
 [0.3, 0.7, -0.2, ...]
 [1.2, 0.1,  0.5, ...]
 [-0.4, 0.9, 0.3, ...]
 [2.1, 1.8, 0.7, ...]
]
```

These vectors contain meaning.

But they contain **no position information**.

---

# Step 2: Create Position Vectors

The original Transformer generates a vector for every possible position.

Position:

```text
0
1
2
3
4
...
```

gets mapped to:

```text
PE(0)
PE(1)
PE(2)
PE(3)
...
```

These vectors are generated mathematically rather than learned.

---

# The Formula

The original paper uses:

For even dimensions:

```math
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
```

For odd dimensions:

```math
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
```

where:

* `pos` = word position
* `i` = embedding dimension index
* `d` = embedding size

---

# Why Sine and Cosine?

Imagine a simple encoding:

```text
Position 0 → [0]
Position 1 → [1]
Position 2 → [2]
Position 3 → [3]
```

This tells us position, but doesn't provide rich structure.

Instead, Transformers create many waves of different frequencies.

For example:

```text
Dimension 0:

0.00
0.84
0.91
0.14
-0.76
...
```

while another dimension changes much more slowly:

```text
Dimension 10:

0.00
0.02
0.04
0.06
0.08
...
```

and another even slower:

```text
Dimension 50:

0.000
0.001
0.002
0.003
...
```

Each dimension acts like a different clock.

Together they create a unique fingerprint for every position.

---

# Visual Intuition

Suppose embedding size is 4.

Position 0:

```text
[0.00, 1.00, 0.00, 1.00]
```

Position 1:

```text
[0.84, 0.54, 0.01, 0.99]
```

Position 2:

```text
[0.91,-0.42, 0.02, 0.99]
```

Position 3:

```text
[0.14,-0.99, 0.03, 0.99]
```

Every position receives a distinct vector.

---

# Step 3: Add Position to Embeddings

Suppose:

```text
movie
```

has embedding:

```text
[1.2, 0.1, 0.5]
```

and position 1 encoding:

```text
[0.84, 0.54, 0.01]
```

Then:

```text
movie_at_position_1
=
[2.04, 0.64, 0.51]
```

because we simply add them.

```python
x = embedding + positional_encoding
```

This is exactly what your code does:

```python
return x + self.pe[:, :seq_len]
```

---

# Why Addition Works

You might wonder:

> Why not concatenate?

For example:

```text
Embedding:
[1.2, 0.1, 0.5]

Position:
[0.8, 0.5, 0.0]
```

could become:

```text
[1.2, 0.1, 0.5, 0.8, 0.5, 0.0]
```

But that doubles dimensionality.

Addition keeps:

```text
embedding_size = constant
```

while still injecting position information.

The network learns to separate semantic meaning from position patterns.

---

# The Hidden Superpower

The sinusoidal design gives something remarkable:

The model can infer relative distances.

For example:

```text
Position 10
Position 11
Position 12
```

have mathematically related vectors.

The attention mechanism can learn:

```text
"look one word back"
```

or

```text
"look three words ahead"
```

because shifts in position correspond to predictable shifts in the sine/cosine waves.

This was one of the motivations for using sinusoidal functions rather than arbitrary numbers.

---

# What Self-Attention Sees

After positional encoding:

```text
"The"    + PE(0)
"movie"  + PE(1)
"was"    + PE(2)
"great"  + PE(3)
```

Self-attention no longer receives just word meanings.

It receives:

```text
meaning + location
```

which allows it to learn patterns such as:

```text
adjective before noun
```

```text
verb after subject
```

```text
word nearby
```

```text
word far away
```

and many other linguistic relationships.

---

# Modern Transformers

The original Transformer used fixed sinusoidal encodings.

Many modern architectures instead learn positional representations:

* BERT uses learned position embeddings.
* GPT-2 uses learned position embeddings.
* LLaMA uses Rotary Positional Embeddings (RoPE).
* Gemma uses RoPE-style positional methods.

However, understanding sinusoidal positional encoding is still the best way to understand the core idea:

> A Transformer knows what a word means from its embedding, and where it appears from its positional encoding. Without positional encoding, self-attention would treat language as an unordered bag of words.

---

## Self-Attention

Self-attention allows each word to determine which other words are important.

Example:

```text
"The movie was incredibly good"
```

The word:

```text
good
```

may learn to focus on:

```text
movie
incredibly
```

rather than unrelated words.

This enables the model to capture long-range relationships within text.

---

## Transformer Encoder Block

The encoder block consists of:

```text
Input
  ↓
Self Attention
  ↓
Residual Connection
  ↓
Feed Forward Network
  ↓
Output
```

Multiple encoder blocks can be stacked to build deeper Transformers.

---

# 📊 Dataset

This project uses the IMDB Movie Review Dataset.

Dataset characteristics:

* 25,000 training reviews
* 25,000 testing reviews
* Binary sentiment classification
* Positive vs Negative reviews

The dataset is automatically downloaded from Stanford's public repository during the first run.

---

# ⚙️ Model Architecture

```text
Input Tokens
      ↓
Embedding Layer
      ↓
Positional Encoding
      ↓
Self Attention
      ↓
Feed Forward Network
      ↓
Mean Pooling
      ↓
Linear Classifier
      ↓
Positive / Negative
```

---

# 🚀 Running the Project

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

# 💾 Model Checkpointing

After training, the model is saved automatically:

```text
outputs/transformer_model.pth
```

Future runs will:

```text
Load existing model
↓
Skip training
↓
Generate visualizations
```

This prevents retraining every time the project is executed.

---

# 📈 Visualizations

The project generates several educational visualizations.

## Training Loss

```text
outputs/training_loss.png
```

Shows model convergence over epochs.

---

## Self-Attention Heatmap

```text
outputs/attention_heatmap.png
```

Visualizes how strongly tokens attend to one another.

---

## Positional Encoding

```text
outputs/positional_encoding.png
```

Displays the sinusoidal position matrix added to embeddings.

---

## Embedding Space

```text
outputs/embedding_space.png
```

Projects learned embeddings into two dimensions using PCA.

Words with similar meanings often cluster together.

---

# 🔍 Example Predictions

Example inputs:

```text
"This movie was absolutely amazing and emotional."
```

Prediction:

```text
Positive
```

---

Input:

```text
"Terrible acting and a boring story."
```

Prediction:

```text
Negative
```

---

# 📚 Key Takeaways

By completing this project, you will understand:

* how text is converted into tokens
* how embeddings represent language
* why positional encoding is necessary
* how self-attention works
* how Transformer encoder blocks are constructed
* how modern NLP architectures differ from RNNs and LSTMs
* how large language models build upon Transformer foundations

---

# 🔬 Future Extensions

Potential improvements:

* Multi-Head Attention
* Layer Normalization
* Residual Connections
* Decoder Architecture
* Autoregressive Text Generation
* GPT-style Language Modeling
* BERT-style Masked Language Modeling
* Transformer Scaling Experiments

---

# References

### Transformer Paper

Attention Is All You Need

Introduced the Transformer architecture and self-attention mechanism.

### Original Attention Mechanism

Neural Machine Translation by Jointly Learning to Align and Translate

Introduced the first modern neural attention mechanism.

### Illustrated Transformer

A highly recommended visual explanation of Transformers:

https://jalammar.github.io/illustrated-transformer/

---

Built as part of a Deep Learning Foundations series focused on implementing modern neural network architectures from first principles.

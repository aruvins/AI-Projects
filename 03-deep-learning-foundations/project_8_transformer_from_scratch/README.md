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

## Positional Encoding

Unlike RNNs and LSTMs, Transformers process all tokens simultaneously. 

Because of this, they need an explicit way to understand word order. 

Sinusoidal positional encodings are added to embeddings:


Embedding + Position Information allowing the model to distinguish:

```text
Dog bites man
```

from:

```text
Man bites dog
```

---

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

This is exactly what the code does:

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

# High-Level Purpose

An encoder block takes a sequence of embeddings:

```text
"The movie was amazing"
```

and transforms them into richer contextual representations.

Initially:

```text
movie = "movie"
```

After several encoder blocks:

```text
movie = "movie in the context of this sentence"
```

The representation becomes increasingly aware of surrounding words.

---

# Full Encoder Architecture

The actual encoder block from the original Transformer paper is:

```text
Input
  ↓
Multi-Head Self Attention
  ↓
Add & Normalize
  ↓
Feed Forward Network
  ↓
Add & Normalize
  ↓
Output
```

---

# Input Shape

Suppose:

```text
Sentence:
"The movie was amazing"
```

Tokenized:

```text
[17, 42, 8, 119]
```

Embedded:

```text
[
 e1,
 e2,
 e3,
 e4
]
```

If:

```text
sequence_length = 4
embedding_size = 128
```

the tensor entering the encoder block has shape:

```text
(4, 128)
```

or in batch form:

```text
(batch_size, 4, 128)
```

---

# Step 1: Self-Attention

This is the most important component.

---

## The Problem

Consider:

```text
"The movie was surprisingly good."
```

To understand:

```text
good
```

the model should pay attention to:

```text
movie
surprisingly
```

rather than every word equally.

---

## Query, Key, Value

For each word embedding:

```text
x
```

the model creates:

```text
Q = xWQ
K = xWK
V = xWV
```

These are learned linear projections.

---

Imagine:

```text
movie
```

becomes:

```text
Qmovie
Kmovie
Vmovie
```

and:

```text
good
```

becomes:

```text
Qgood
Kgood
Vgood
```

---

## Computing Attention

The model asks:

> "How much should word A pay attention to word B?"

Using:

Attention(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

---

### Example

Suppose:

```text
good
```

looks at:

```text
the
movie
was
surprisingly
good
```

Attention scores might become:

```text
[
 0.05,
 0.30,
 0.10,
 0.35,
 0.20
]
```

Meaning:

```text
5%   → the
30%  → movie
10%  → was
35%  → surprisingly
20%  → itself
```

These weights sum to:

```text
1.0
```

---

## Result

The word:

```text
good
```

gets replaced by:

```text
weighted combination
of all words
```

creating a context-aware representation.

---

# Step 2: Residual Connection

After attention:

```text
attention_output
```

we add the original input:

```text
input + attention_output
```

Diagram:

```text
Input --------+
              |
              v
        Self Attention
              |
              +
              |
           Output
```

---

## Why?

Deep networks often suffer from:

```text
vanishing gradients
```

Residual connections provide a shortcut.

Instead of learning:

```text
Output
```

the layer learns:

```text
Difference from Input
```

This dramatically improves training stability.

This idea originated in the paper introducing Deep Residual Learning for Image Recognition.

---

# Step 3: Layer Normalization

Usually:

```python
x = LayerNorm(
    x + attention_output
)
```

---

## Why?

Different features may have wildly different scales.

Example:

```text
Feature 1 = 0.2
Feature 2 = 500
Feature 3 = -20
```

Training becomes unstable.

LayerNorm rescales activations:

```text
mean ≈ 0
std ≈ 1
```

making optimization easier.

---

# Step 4: Feed Forward Network

Many beginners assume attention is where all learning happens.

Actually, attention mostly moves information around.

The feed-forward network performs substantial computation.

Typically:

```text
128
 ↓
512
 ↓
128
```

or:

```python
Linear(128,512)
ReLU()
Linear(512,128)
```

---

## Why Expand?

Suppose the attention layer creates:

```text
movie representation
```

The FFN can then learn:

```text
sentiment
genre
emotion
negation
sarcasm
```

and other complex features.

---

Diagram:

```text
128 dims
    ↓
512 dims
    ↓
128 dims
```

This gives the network more expressive power.

---

# Step 5: Second Residual Connection

Again:

```python
output = x + ffn_output
```

Diagram:

```text
Input
  ↓
Attention
  ↓
Add
  ↓
FFN
  ↓
Add
  ↓
Output
```

Now both major computations have shortcut paths.

---

# What Happens Across Multiple Layers?

One encoder block is useful.

Multiple blocks are powerful.

Suppose:

```text
"The stock crashed after earnings."
```

---

### Encoder Block 1

Learns:

```text
stock
earnings
crashed
```

are related.

---

### Encoder Block 2

Learns:

```text
crashed → negative event
```

---

### Encoder Block 3

Learns:

```text
earnings caused crash
```

---

### Encoder Block 4

Builds a richer semantic understanding.

Each layer increases abstraction.

This is similar to CNNs:

```text
Edges
 ↓
Shapes
 ↓
Objects
```

except for language.

---

# Why Stacking Works

Each block receives:

```text
contextualized representations
```

from the previous block.

Not raw words.

So deeper layers can reason about increasingly complex relationships.

---

# Shape Tracking Example

Assume:

```text
batch_size = 64
sequence_length = 200
embedding_size = 128
```

Input:

```text
(64, 200, 128)
```

After attention:

```text
(64, 200, 128)
```

After residual:

```text
(64, 200, 128)
```

After feed-forward:

```text
(64, 200, 128)
```

Output:

```text
(64, 200, 128)
```

Notice:

**the shape never changes.**

This is why encoder blocks can be stacked repeatedly:

```text
Encoder 1
 ↓
Encoder 2
 ↓
Encoder 3
 ↓
Encoder 4
```

Every block accepts and returns the same tensor dimensions.

---

# Intuition

A useful mental model is:

```text
Embeddings
    ↓
"What do words mean?"
```

```text
Self-Attention
    ↓
"Which words matter to each other?"
```

```text
Feed Forward Network
    ↓
"What higher-level features can I learn?"
```

```text
Residual Connections
    ↓
"Keep training stable."
```

Stack enough encoder blocks together and you get the foundation of models like BERT. The decoder architecture used by models such as GPT-2 uses the same core ideas but adds causal masking so the model can generate text one token at a time.

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

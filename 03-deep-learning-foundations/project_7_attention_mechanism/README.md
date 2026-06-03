# 🧠 Project 7 — Attention Mechanism Playground

## 🎯 Goal

Learn how **attention works** before learning full Transformer architectures.

This project implements a simplified version of **self-attention** from scratch using PyTorch and visualizes every major step in the attention computation.

By the end of this project, you will understand:

* What attention is
* Why attention replaced RNNs and LSTMs for many NLP tasks
* What Queries, Keys, and Values are
* How attention scores are computed
* Why Softmax is used
* How context vectors are created
* The core computation that powers modern Transformer models such as GPT, BERT, and Vision Transformers

---

# Project Structure

```text
project_7_attention_mechanism/
│
├── models/
│   └── attention.py
│
├── utils/
│   ├── tokenizer.py
│   ├── visualization.py
│   └── examples.py
│
├── outputs/
│   ├── attention_heatmap.png
│   ├── raw_scores.png
│   ├── query_vectors.png
│   ├── key_vectors.png
│   └── value_vectors.png
│
├── main.py
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_7_attention_mechanism

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

# Why This Project Matters

Before Transformers became dominant, most language models used:

* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory Networks (LSTMs)

These models process text sequentially:

```text
Word 1 → Word 2 → Word 3 → Word 4
```

As sentences become longer, it becomes difficult for information from earlier words to influence later words.

For example:

```text
The cat that lived across the street from the old house sat on the mat.
```

To understand the word:

```text
sat
```

the model may need information from:

```text
cat
```

which appeared much earlier in the sentence.

Attention solves this problem by allowing every word to directly examine every other word.

---

# What Is Attention?

Imagine reading the sentence:

```text
The cat sat on the mat
```

When processing the word:

```text
cat
```

you naturally pay attention to nearby words:

```text
the
cat
sat
```

because they provide useful context.

Attention allows a neural network to do the same thing.

Instead of processing words one at a time, the model asks:

```text
Which words are most important for understanding this word?
```

---

## Visual Intuition

```text
             cat
              │
      ┌───────┼───────┐
      │       │       │
     the     sat     mat
      │       │       │
      └───────┼───────┘
              │
       Gather Context
```

The model learns how much attention each word should receive.

---

# The Core Attention Equation

Self-attention is defined as:

```text
Attention(Q, K, V)
=
Softmax(QKᵀ / √dₖ)V
```

Although this equation looks intimidating, it can be broken into simple steps.

---

# Understanding Queries, Keys, and Values

The attention mechanism creates three different representations of each word.

## Query

A Query asks:

```text
"What information am I looking for?"
```

---

## Key

A Key answers:

```text
"What information do I contain?"
```

---

## Value

A Value contains:

```text
"The information I can contribute."
```

---

## Analogy

Imagine a search engine.

### Query

```text
"best pizza in NYC"
```

### Keys

Every webpage contains metadata describing its content.

### Values

The actual webpage content.

The search engine compares:

```text
Query vs Key
```

to determine relevance.

Then it gathers information from the most relevant Values.

Attention works in exactly the same way.

---

# Attention Pipeline

The complete workflow looks like this:

```text
Input Sentence
        │
        ▼

Word Embeddings
        │
        ▼

        X
        │

 ┌──────┼──────┐
 ▼      ▼      ▼

 Q      K      V

 │      │
 └──┬───┘
    ▼

  QKᵀ

    ▼

 Attention Scores

    ▼

 Scaling

    ▼

 Softmax

    ▼

 Attention Weights

    ▼

 Attention × V

    ▼

 Context Vectors
```

---

# Step 1 — Tokenization

Sentence:

```text
The cat sat on the mat
```

becomes:

```python
["the", "cat", "sat", "on", "the", "mat"]
```

---

# Step 2 — Embeddings

Each token is converted into a vector.

Example:

```text
cat
=
[0.73, -0.21, 1.34, ...]
```

The embedding represents semantic information about the word.

Words with similar meanings tend to have similar embeddings.

---

# Step 3 — Generate Queries, Keys, and Values

The model applies three learned linear layers:

```python
Q = self.query(x)
K = self.key(x)
V = self.value(x)
```

These layers learn how words should search for and provide information.

---

# Step 4 — Compute Attention Scores

The model computes:

```python
scores = torch.matmul(Q, K.T)
```

This compares every Query against every Key.

Example:

```text
          the   cat   sat

the      0.3   1.1   0.7

cat      0.4   2.0   1.5

sat      0.2   1.2   1.8
```

Each row answers:

```text
How much attention should this word pay
to every other word?
```

---

# Step 5 — Scale Scores

The scores are divided by:

```python
sqrt(d_k)
```

where:

```text
d_k
=
key dimension
```

This prevents large values from causing unstable gradients.

---

# Step 6 — Softmax

The scores become probabilities.

Before:

```text
[0.4, 2.0, 1.5]
```

After:

```text
[0.11, 0.55, 0.34]
```

Interpretation:

```text
11% attention → the
55% attention → cat
34% attention → sat
```

Every row sums to:

```text
1.0
```

---

# Step 7 — Build Context Vectors

The attention weights are used to combine Value vectors.

```python
context =
    torch.matmul(
        attention_weights,
        V
    )
```

The resulting context vector becomes a richer representation of each word.

Instead of representing only:

```text
cat
```

the new representation contains information gathered from:

```text
the
cat
sat
mat
```

and every other word in the sentence.

---

# Understanding the Heatmap

The project generates:

```text
outputs/attention_heatmap.png
```

Example:

```text
          the   cat   sat

the      0.20  0.50  0.30

cat      0.10  0.70  0.20

sat      0.15  0.25  0.60
```

Interpretation:

```text
cat → cat = 70%

cat → sat = 20%

cat → the = 10%
```

Brighter cells indicate stronger attention.

The heatmap is one of the most useful ways to understand how attention behaves.

---

# Why Attention Changed AI

Attention provides several major advantages.

## Parallel Processing

RNNs process tokens sequentially:

```text
Word 1
↓
Word 2
↓
Word 3
```

Attention processes all words simultaneously.

---

## Long-Range Dependencies

Attention can directly connect:

```text
Word 1
```

with

```text
Word 100
```

without passing through 98 intermediate states.

---

## Better Scaling

Attention scales more effectively to large datasets and larger models.

This is one of the main reasons Transformers surpassed RNNs and LSTMs.

---

# Suggested Experiments

Try modifying:

## Embedding Dimension

```python
embedding_dim=8
```

Try:

```python
16
32
64
128
```

and observe how attention changes.

---

## Different Sentences

Try:

```text
attention helps models focus
```

or

```text
the dog chased the cat
```

and compare heatmaps.

---

## Multiple Attention Heads

After understanding this project, implement:

```text
Multi-Head Attention
```

to learn how Transformers combine multiple attention patterns.

---

# Next Projects

After completing this project, a good learning path is:

```text
Project 1
Perceptron

Project 2
MNIST Classifier

Project 3
Fashion-MNIST

Project 4
CNN on CIFAR-10

Project 5
RNN

Project 6
LSTM

Project 7
Attention Playground  ← You are here

Project 8
Transformer Encoder

Project 9
Mini GPT
```

# References

## Original Attention Paper

Bahdanau, Cho, and Bengio (2014)

**Neural Machine Translation by Jointly Learning to Align and Translate**

Paper:
https://arxiv.org/abs/1409.0473

This paper introduced the first widely adopted neural attention mechanism and showed how models could learn to focus on relevant parts of an input sequence.

---

## Transformer Paper

Vaswani et al. (2017)

**Attention Is All You Need**

Paper:
https://arxiv.org/abs/1706.03762

The paper that introduced the Transformer architecture and scaled self-attention, laying the foundation for modern large language models.

---

## Deep Learning Textbook

Ian Goodfellow, Yoshua Bengio, Aaron Courville

**Deep Learning**

Free Online Book:
https://www.deeplearningbook.org/

Recommended Chapters:

* Chapter 6 — Deep Feedforward Networks
* Chapter 10 — Sequence Modeling
* Chapter 12 — Applications

A comprehensive resource for understanding the theoretical foundations behind neural networks and sequence models.

---

## The Illustrated Transformer

Jay Alammar

Article:
https://jalammar.github.io/illustrated-transformer/

Article 2:
https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

One of the best visual explanations of attention and Transformers available online. Highly recommended before reading the original Transformer paper.

---

## Stanford CS224N

**Natural Language Processing with Deep Learning**

Course Website:
https://web.stanford.edu/class/cs224n/

Lecture Videos:
https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ

Excellent university-level course covering:

* Word Embeddings
* RNNs
* LSTMs
* Attention
* Transformers
* Large Language Models

---

## Annotated Transformer

Harvard NLP

Article:
https://nlp.seas.harvard.edu/annotated-transformer/

A line-by-line implementation of the Transformer architecture in PyTorch with detailed explanations.

This is often considered the best bridge between understanding attention and implementing a Transformer from scratch.

---

## 3Blue1Brown Neural Networks Series

YouTube Playlist:
https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr

Provides outstanding intuition for:

* Vectors
* Matrices
* Gradients
* Backpropagation

Helpful background knowledge before studying attention in depth.

---

## Andrej Karpathy

Neural Networks: Zero to Hero

Playlist:
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

Particularly recommended:

* Micrograd
* Makemore
* GPT from Scratch

Excellent preparation for implementing Transformers and language models.

---

## Hugging Face Course

Course:
https://huggingface.co/course

Covers:

* Transformers
* Tokenization
* Attention Mechanisms
* Fine-Tuning
* Large Language Models

Provides practical experience using modern NLP tooling.

---

# Suggested Reading Order

For beginners:

1. The Illustrated Transformer
2. Stanford CS224N Attention Lectures
3. Attention Is All You Need
4. The Annotated Transformer
5. Hugging Face Course
6. GPT from Scratch (Karpathy)

This progression moves from intuition → mathematics → implementation → real-world applications.

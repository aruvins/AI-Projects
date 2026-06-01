🟡 Project 6 — Text Classification with RNN/LSTM
🎯 Goal

Learn how neural networks process sequential data such as text, sentences, and time-series information.

Unlike traditional feedforward neural networks, sequence models maintain memory of previous inputs, allowing them to understand context and relationships across time. This project introduces Recurrent Neural Networks (RNNs) and Long Short-Term Memory Networks (LSTMs) through a text classification task.

The project trains both architectures on the same sentiment analysis dataset and compares their performance.

---

# Project Goals

The objective of this project is to learn:

* Text preprocessing
* Vocabulary construction
* Tokenization
* Word embeddings
* Sequence modeling
* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory Networks (LSTMs)
* Neural network training in PyTorch
* Model evaluation and comparison

The project trains both an RNN and an LSTM on sentiment classification and compares their performance through visualizations.

---

# Project Structure

```text
project_6_text_classification_rnn/
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── rnn_model.py
│   └── lstm_model.py
│
├── utils/
│   ├── vocabulary.py
│   ├── trainer.py
│   ├── predictor.py
│   └── visualize.py
│
├── outputs/
│   ├── loss_comparison.png
│   ├── accuracy_comparison.png
│   └── predictions.txt
│
├── requirements.txt
├── main.py
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_6_text_classification_rnn

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

## Train the CNN and visualize

```bash
python main.py
```

During execution the program will:

1. Build a vocabulary from the training data
2. Convert text into numerical sequences
3. Train an RNN classifier
4. Train an LSTM classifier
5. Compare training performance
6. Generate visualizations
7. Save predictions to a text file

---

# Generated Outputs

After training, the outputs folder will contain:

```text
outputs/

loss_comparison.png
accuracy_comparison.png
predictions.txt
```

## Loss Comparison

Shows how training loss changes over time for both models.

Lower loss generally indicates better learning.

## Accuracy Comparison

Shows how classification accuracy improves throughout training.

## Predictions File

Contains predictions from both models on custom review samples.

Example:

```text
Sentence: This movie was incredible
RNN Prediction : Positive
LSTM Prediction: Positive
```

---

# NLP Pipeline

The overall workflow looks like this:

```text
Raw Text
    ↓
Tokenization
    ↓
Vocabulary
    ↓
Integer Encoding
    ↓
Embedding Layer
    ↓
RNN / LSTM
    ↓
Fully Connected Layer
    ↓
Sentiment Prediction
```

Example:

```text
"I loved this movie"

↓ Tokenization

["i", "loved", "this", "movie"]

↓ Encoding

[15, 87, 4, 22]

↓ Embedding

Dense Vectors

↓ LSTM

Hidden Representation

↓ Classifier

Positive
```

---

# Understanding Integer Encoding and Embeddings

Before a neural network can process text, words must be converted into numbers.

This transformation happens in two major stages:

```text
Text
 ↓
Integer Encoding
 ↓
Embedding Layer
 ↓
Dense Vector Representations
```

Although these steps seem simple, they are fundamental to nearly every modern Natural Language Processing (NLP) system.

---

# Step 1 — Building a Vocabulary

Suppose we have the following sentences:

```text
I loved this movie
This movie was fantastic
The acting was terrible
```

The first step is to collect every unique word.

```text
i
loved
this
movie
was
fantastic
the
acting
terrible
```

We then assign each word a unique integer ID.

```text
<pad>      → 0
<unk>      → 1

i          → 2
loved      → 3
this       → 4
movie      → 5
was        → 6
fantastic  → 7
the        → 8
acting     → 9
terrible   → 10
```

This mapping is called the vocabulary.

---

# Step 2 — Integer Encoding

Now every word can be replaced by its corresponding ID.

Original sentence:

```text
I loved this movie
```

Encoded sentence:

```text
[2, 3, 4, 5]
```

Another example:

```text
The acting was terrible
```

becomes:

```text
[8, 9, 6, 10]
```

At this point the computer can store and manipulate the text numerically.

However, there is still a major problem.

---

# Why Integer IDs Are Not Enough

Consider:

```text
great → 15
excellent → 42
terrible → 8
```

The neural network sees:

```text
15
42
8
```

These numbers do not contain any information about meaning.

The model cannot infer:

```text
great ≈ excellent
```

or

```text
great ≠ terrible
```

because the assigned IDs are completely arbitrary.

For example:

```text
great → 15
excellent → 16
terrible → 17
```

does NOT mean:

```text
excellent is closer to great
than terrible
```

The numbers are simply labels.

This is why we need embeddings.

---

# Step 3 — The Embedding Layer

An embedding layer converts each integer ID into a learned vector.

Instead of:

```text
great → 15
```

the network learns:

```text
great →

[
  0.42,
 -0.81,
  1.14,
  ...
]
```

If the embedding dimension is 128:

```text
great → 128 numbers
```

If the embedding dimension is 300:

```text
great → 300 numbers
```

Each word becomes a point in a high-dimensional space.

---

# What Does the Embedding Layer Actually Store?

Inside the embedding layer is a large matrix.

Suppose:

```text
Vocabulary Size = 10,000
Embedding Dimension = 128
```

Then the embedding matrix has shape:

```text
(10000, 128)
```

Every row corresponds to one word.

```text
Word ID 0 → [ ... ]
Word ID 1 → [ ... ]
Word ID 2 → [ ... ]
Word ID 3 → [ ... ]
...
```

Visually:

```text
Embedding Matrix

        128 Features
      ┌───────────────┐
ID 0  │ x x x x x x x │
ID 1  │ x x x x x x x │
ID 2  │ x x x x x x x │
ID 3  │ x x x x x x x │
      └───────────────┘
```

Each row is a learned word representation.

---

# How nn.Embedding Works

In PyTorch:

```python
self.embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=128
)
```

When the input is:

```python
x = torch.tensor([
    [2, 3, 4, 5]
])
```

PyTorch performs a lookup.

Conceptually:

```python
embedding_matrix[2]
embedding_matrix[3]
embedding_matrix[4]
embedding_matrix[5]
```

The output becomes:

```python
[
    [128 values],
    [128 values],
    [128 values],
    [128 values]
]
```

Resulting shape:

```text
(batch_size, sequence_length, embedding_dim)

(1, 4, 128)
```

---

# Example Inside the Network

Input sentence:

```text
I loved this movie
```

Integer encoded:

```text
[2, 3, 4, 5]
```

Embedding output:

```text
[
 [0.12, -0.44, ...],
 [0.91,  0.22, ...],
 [-0.30, 0.17, ...],
 [0.65, -1.03, ...]
]
```

These vectors are then passed into the RNN or LSTM.

```text
Encoded Tokens
        ↓
Embedding Layer
        ↓
Word Vectors
        ↓
RNN / LSTM
        ↓
Prediction
```

---

# How Are Embeddings Learned?

Initially the embedding values are random.

For example:

```text
great

[0.11, -0.27, 0.08]
```

During training:

```text
Forward Pass
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Embedding Update
```

The embedding vectors are adjusted along with the rest of the network.

Eventually:

```text
great
excellent
fantastic
```

receive similar vectors because they help the model make correct predictions.

---

# Semantic Relationships

One of the most interesting properties of embeddings is that they often capture meaning.

After training, vectors may look like:

```text
great
excellent
fantastic
```

cluster together.

While:

```text
terrible
awful
horrible
```

form another cluster.

Visualized:

```text
Positive Words

great
excellent
fantastic
      ● ● ●

Negative Words

terrible
awful
horrible
      ○ ○ ○
```

Words with similar meanings tend to occupy nearby regions of embedding space.

---

# Why Embeddings Are Powerful

Without embeddings:

```text
great → 15
excellent → 42
```

The network sees unrelated numbers.

With embeddings:

```text
great     → [0.8, 0.3, -0.1]
excellent → [0.7, 0.4, -0.2]
```

The network can recognize that the words are semantically similar.

This dramatically improves learning and is one of the key innovations that enabled modern NLP systems.

---

# Summary

Integer encoding and embeddings serve different purposes:

### Integer Encoding

```text
Word
 ↓
Unique ID
```

Example:

```text
movie → 5
```

Purpose:

```text
Convert text into discrete numerical tokens
```

---

### Embedding Layer

```text
Word ID
 ↓
Dense Vector
```

Example:

```text
5 →

[0.42, -0.18, 1.21, ...]
```

Purpose:

```text
Learn semantic representations of words
```

The embedding layer transforms arbitrary word IDs into meaningful numerical representations that can be understood by sequence models such as RNNs and LSTMs.

---

# What is Sequence Modeling?

Unlike image classification, text contains order.

These two sentences contain the same words:

```text
bad movie not good
```

```text
good movie not bad
```

but mean different things.

Sequence models process information in order and learn relationships between words over time.

---

# Recurrent Neural Networks (RNNs)

Traditional neural networks assume all inputs are independent.

Text is different because previous words influence future words.

RNNs introduce a hidden state that acts as memory.

At each step:

```text
Current Word
      +
Previous Memory
      ↓
 New Memory
```

The hidden state is updated repeatedly as the sequence is processed.

Example:

```text
I
↓
loved
↓
this
↓
movie
```

The network reads one word at a time and continuously updates its internal memory.

---

# Hidden States

The hidden state stores information learned from previous words.

```text
Word 1
↓
Hidden State 1

Word 2
↓
Hidden State 2

Word 3
↓
Hidden State 3
```

This allows the model to capture context.

For example:

```text
The movie was not good
```

When processing:

```text
good
```

the model can remember:

```text
not
```

from earlier in the sentence.

---

# The Vanishing Gradient Problem

RNNs struggle with long sequences.

During backpropagation, gradients repeatedly pass through many time steps.

They can become extremely small:

```text
0.5 × 0.5 × 0.5 × 0.5 × ...
```

Eventually:

```text
≈ 0
```

When gradients vanish, the network stops learning long-range relationships.

This is known as the vanishing gradient problem.

---

# Long Short-Term Memory Networks (LSTMs)

LSTMs were designed specifically to solve the limitations of vanilla RNNs.

Instead of a simple hidden state, LSTMs maintain:

```text
Hidden State
+
Cell State
```

The cell state acts as long-term memory.

This allows information to travel through many time steps with minimal loss.

---

# LSTM Architecture

An LSTM contains three gates:

```text
Forget Gate
Input Gate
Output Gate
```

These gates control information flow.

---

## Forget Gate

Determines what information should be removed from memory.

```text
Important?
    ↓
Keep

Not Important?
    ↓
Forget
```

---

## Input Gate

Determines what new information should be stored.

```text
New Information
      ↓
Store in Memory
```

---

## Output Gate

Determines what information should be exposed to the next layer.

```text
Memory
   ↓
Useful Information
   ↓
Output
```

---

# Why LSTMs Perform Better

Consider:

```text
The movie started slowly and felt boring at times,
but the ending was absolutely fantastic.
```

A vanilla RNN may forget:

```text
started slowly
```

before reaching:

```text
fantastic
```

An LSTM can maintain information across many words and better understand long-range dependencies.

This makes LSTMs significantly more effective for many NLP tasks.

---

# RNN vs LSTM

| Feature                       | RNN   | LSTM   |
| ----------------------------- | ----- | ------ |
| Sequential Processing         | ✓     | ✓      |
| Hidden State                  | ✓     | ✓      |
| Long-Term Memory              | ✗     | ✓      |
| Vanishing Gradient Resistance | Poor  | Strong |
| Performance on Long Sequences | Lower | Higher |
| Computational Cost            | Lower | Higher |

---

# Training Process

For each batch:

```text
Forward Pass
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Weight Update
```

The optimizer used is:

```python
optim.Adam()
```

Adam adapts learning rates automatically and is one of the most widely used optimizers in deep learning.

---

# What This Project Demonstrates

By completing this project you will understand:

* How text is converted into numerical representations
* How embeddings work
* How sequence models process text
* How hidden states store information
* Why RNNs struggle with long-term dependencies
* How LSTMs solve the vanishing gradient problem
* How to train NLP models in PyTorch
* How to compare multiple neural network architectures

---

# Future Improvements

Possible extensions include:

* Larger sentiment datasets
* Bidirectional LSTMs
* GRUs (Gated Recurrent Units)
* Attention mechanisms
* Transformer encoders
* BERT fine-tuning
* Real-world review datasets
* Pretrained word embeddings (GloVe, Word2Vec)

---

# Next Project

After understanding RNNs and LSTMs, the next natural step is:

```text
Transformer Text Classification
```

where recurrence is replaced by self-attention.

Understanding RNNs and LSTMs first provides valuable intuition for why Transformers became the dominant architecture in modern Natural Language Processing.

# 🟡 Project 6 — Text Classification with RNN/LSTM

## 🎯 Goal

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

## Side Note on Special Tokens

Special Tokens

```text
<pad> (Padding Token)
```
Used to make all sequences the same length.

Neural networks process data in batches, and each batch requires uniform tensor shapes.

Example:
```text
Sentence A: I loved this movie
Sentence B: Amazing film
```

After padding:
```text
I loved this movie
Amazing film <pad> <pad>
```

Purpose:
* Ensures consistent sequence length
* Allows batching during training
* Has no semantic meaning

---

```text
<unk> (Unknown Token)
```

Used for words that are not in the vocabulary.

Example:
```text
"I loved this cinematography"

If "cinematography" is not in the vocabulary:

"I loved this <unk>"
```

Purpose:
* Handles unseen or rare words
* Prevents model failure on out-of-vocabulary inputs
* Acts as a fallback representation

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
...

Negative Words

terrible
awful
horrible
...
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

# Summary on Encodings and Embeddings

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

but mean completely different things.

Sequence models process information in order and learn relationships between words over time.

---

## Why Order Matters

In NLP, meaning is not only in the words — it is in how they are arranged.

For example:

```text
not good
```

is negative,

but:

```text
good not
```

is grammatically broken and semantically unclear.

This means the model must understand **structure**, not just vocabulary.

---

## Fixed-Length Models Fail

Traditional neural networks (like simple feedforward networks) treat inputs as independent.

They assume:

```text
word1, word2, word3 are unrelated inputs
```

So:

```text
bad movie not good
```

and

```text
good movie not bad
```

would look almost identical if you just average or sum embeddings.

This is a major limitation.

---

## The Key Idea Behind Sequence Models

Sequence models introduce the concept of **memory**.

Instead of processing all words at once, they read text step by step:

```text
Word 1 → Word 2 → Word 3 → Word 4
```

At each step, the model updates its internal understanding.

---

## Step-by-Step Intuition

Consider the sentence:

```text
The movie was not good
```

The model processes it like this:

```text
Step 1: "The" → start context
Step 2: "movie" → update context (topic = film)
Step 3: "was" → grammatical connector
Step 4: "not" → introduces negation (VERY important signal)
Step 5: "good" → final sentiment word
```

At the end, the model does NOT just see “good”.

It sees:

```text
good + not → negative sentiment
```

---

## Hidden State = Memory

The core of sequence modeling is the **hidden state**.

At each time step:

```math
h_t = f(x_t, h_{t-1})
```

Where:

* `x_t` = current word
* `h_{t-1}` = previous memory
* `h_t` = updated memory

So the model is constantly asking:

```text
"What have I seen so far?"
```

---

## Information Accumulates Over Time

As words are processed, meaning builds up gradually.

Example:

```text
"I really did not expect this movie to be so good"
```

Early words:

```text
"I really did not expect"
→ neutral / uncertain context
```

Later words:

```text
"so good"
→ strong positive signal
```

Final prediction depends on **entire sequence history**, not just last word.

---

## Direction of Processing

Most RNNs and LSTMs process sequences in one direction:

```text
Left → Right
```

So they read:

```text
bad → movie → not → good
```

Each word influences the next hidden state.

This is called **autoregressive processing**.

---

## Why This Is Powerful

Sequence modeling allows networks to learn:

* word order importance
* negation (“not good” vs “good”)
* context dependency
* long-range relationships
* sentence structure patterns

---

## The Problem with Long Sequences

As sequences get longer:

```text
The movie started slowly, the acting was fine, the story was okay, the pacing was a bit off, but the ending was amazing
```

The model must remember:

```text
"ending was amazing"
```

while not forgetting earlier context.

This becomes difficult for vanilla RNNs.

---

## Why Memory Matters

Without memory:

```text
"not good"
→ model sees only "good"
→ wrong prediction
```

With memory:

```text
model remembers "not"
→ flips meaning of "good"
→ correct prediction
```

---

## Sequence Models Learn Patterns

Instead of memorizing sentences, they learn patterns like:

```text
"not + positive word → negative sentiment"
```

or

```text
"very + good → strong positive sentiment"
```

This is called **representation learning over time**.

---

## Temporal Dependency

A key idea:

```text
Earlier words affect later understanding
```

This is called **temporal dependency**.

Example:

```text
Although the movie was boring at first, ...
```

The word “although” signals:

```text
expect contrast later
```

So the model anticipates a shift in sentiment.

---

## Summary Intuition

Sequence modeling is about teaching a neural network to:

* read step-by-step
* maintain memory
* update understanding over time
* interpret meaning based on context
* handle word order sensitivity

---

## Connection to RNNs and LSTMs

RNNs implement sequence modeling using:

```text
simple hidden memory
```

LSTMs improve this by adding:

```text
selective memory (gates)
```

This allows them to decide:

* what to remember
* what to forget
* what to emphasize

---

## Big Picture

Sequence models answer one key question:

```text
How do we give neural networks memory so they can understand ordered data like language?
```

This idea is the foundation for:

* RNNs
* LSTMs
* GRUs
* Transformers (modern evolution)

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

## How Hidden States Actually Work

A hidden state is essentially the model’s **internal memory vector** at each time step.

It is not a single value — it is a vector of numbers, for example:

```text
h_t = [0.12, -0.44, 0.87, ...]
```

This vector stores compressed information about everything seen so far in the sequence.

---

At each step, the RNN combines two things:

```text
Current input (word embedding)
+
Previous hidden state (memory)
```

and produces a new hidden state.

---

Mathematically:

```math
h_t = tanh(W*x_t + U*h_{t-1} + b)
```

Where:

* `x_t` = current word embedding
* `h_{t-1}` = previous memory
* `W` = weights for input
* `U` = weights for memory
* `b` = bias term

---

## Step-by-Step Intuition

Let’s break it down:

### Step 1: First word

```text
"I"
```

The model has no prior context, so:

```text
h_1 = f("I")
```

Memory now contains basic sentence context.

---

### Step 2: Second word

```text
"movie"
```

Now the model combines:

```text
h_2 = f("movie", h_1)
```

So it updates memory to represent:

* subject = sentence is about a movie

---

### Step 3: Third word

```text
"not"
```

Now this becomes important:

```text
h_3 = f("not", h_2)
```

The hidden state now stores a **negation signal**, even if indirectly.

---

### Step 4: Fourth word

```text
"good"
```

Now the model processes:

```text
h_4 = f("good", h_3)
```

Even though "good" is positive on its own, the hidden state already contains “not”.

So the final meaning becomes:

```text
NEGATIVE sentiment
```

---

## Why This Feels Like Memory

The hidden state acts like a rolling summary of the sentence.

Instead of storing all words, the model compresses them into a vector:

```text
All previous words → single memory vector
```

This is why it is called a **recurrent** network — it repeatedly reuses its own output.

---

## What the Hidden State Learns

Over training, hidden states start encoding patterns like:

* sentiment direction (positive/negative)
* sentence topic (movie, product, review)
* negation structure (“not”, “never”, “no”)
* intensity (“very”, “extremely”)
* word dependencies across time

---

## Important Limitation

While hidden states store memory, they are not perfect.

As sequences get longer:

```text
memory becomes compressed repeatedly
```

This can cause earlier information to fade away.

For example:

```text
"I really loved the beginning of the movie which was slow but interesting..."
```

By the end, early details may be weakened.

---

## Why This Leads to LSTMs

This limitation is called the **vanishing memory problem**.

It is one of the main reasons LSTMs were introduced — they improve how information is preserved over long sequences using a dedicated cell state.

---

## Key Intuition

A hidden state is:

* a summary of everything seen so far
* updated at every word
* reused for the next prediction
* the core “memory mechanism” of RNNs

Without hidden states, an RNN would have no way to remember earlier words in a sentence.

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

```text id="kq9x8c"
Forget Gate
Input Gate
Output Gate
```

These gates control information flow.

---

## Forget Gate

Determines what information should be removed from memory.

```text id="1v8qmw"
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

```text id="x7p2ld"
New Information
      ↓
Store in Memory
```

---

## Output Gate

Determines what information should be exposed to the next layer.

```text id="8m3f0a"
Memory
   ↓
Useful Information
   ↓
Output
```

---

## How the LSTM Decides What Each Gate Should Do

Each gate is not manually programmed — it is learned during training.

Every gate is actually a small neural network that outputs values between 0 and 1 using a sigmoid function.

```text id="q9l2sp"
0 → block completely
1 → allow fully
```

So instead of rules like “if word = not then forget”, the model learns soft decisions.

---

## Gate Computation (Core Idea)

Each gate looks at:

```text id="b2m7vx"
Current word + Previous hidden state
```

and produces a decision.

For example:

```math id="k8v9ds"
f_t = σ * (W_f * [h_{t-1}, x_t] + b_f)
```

Where:

* `f_t` = forget gate output
* `σ` = sigmoid function
* `h_{t-1}` = previous hidden state
* `x_t` = current word embedding

---

## Forget Gate in Practice

The forget gate decides what parts of old memory are no longer useful.

Example sentence:

```text id="t7p4lm"
The movie was not good
```

When the model sees “not”, it may learn to:

* reduce importance of earlier positive signals
* keep negation-related context active

So memory is continuously “filtered”.

---

## Input Gate Decision

The input gate decides what new information is worth storing.

It works in two steps:

```text id="m3x9qv"
Step 1: Decide what to update
Step 2: Create candidate information
```

This means the model does not blindly store every word.

For example:

```text id="c8n1pl"
"movie"
```

may be stored strongly because it defines topic.

But:

```text id="h4q7zt"
"the"
```

may be mostly ignored.

---

## Candidate Memory Update

The LSTM also generates a candidate memory vector:

```math id="z6p1wa"
~C_t = tanh(W_c [h_{t-1}, x_t] + b_c)
```

This represents “new information the model might add”.

The input gate then decides how much of it is used.

---

## Output Gate Decision

The output gate controls what part of memory becomes visible.

Even if memory contains a lot of information, not all of it is exposed.

```text id="v2k9sd"
Memory → filtered → output
```

This helps the model focus on relevant context for the current prediction.

---

## How Everything Works Together

At each time step:

```text id="p1x8nv"
1. Forget gate removes irrelevant old memory
2. Input gate adds new relevant information
3. Cell state is updated
4. Output gate decides what to expose
```

---

## Why This Is Powerful

Instead of storing a single compressed memory like an RNN, LSTMs maintain:

```text id="w9c2kx"
Long-term memory (cell state)
+ 
Short-term focus (hidden state)
```

This separation allows the model to:

* remember early words in long sentences
* ignore irrelevant noise
* focus on important signals like negation or sentiment words

---

## Example Intuition

Sentence:

```text id="r8m0dp"
Although the movie started slow, it became amazing later
```

The LSTM learns:

* early words → low importance (forget gate reduces weight)
* “amazing” → high importance (input gate increases storage)
* final output → strong positive sentiment (output gate emphasizes key signal)

---

## Key Insight

The model does NOT explicitly understand grammar.

Instead, it learns:

```text id="y4n6qp"
Which information helps reduce prediction error
```

So the gates become learned decision-makers that continuously optimize memory flow.

---

## Summary

LSTM gates are:

* **Forget Gate** → removes unnecessary memory
* **Input Gate** → adds new useful information
* **Output Gate** → controls what is revealed

Each gate is a learned function, not a rule-based system.

Together, they allow LSTMs to selectively remember important information over long sequences while filtering out noise, which is why they outperform vanilla RNNs on complex language tasks.

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

---

# 📚 References

## 🧠 Deep Learning & NLP Fundamentals

* [Understanding LSTM Networks (Colah’s Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/?utm_source=chatgpt.com)
  One of the most intuitive explanations of LSTM architecture, gates, and memory flow.

* [PyTorch RNN Documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?utm_source=chatgpt.com)
  Official PyTorch documentation for Recurrent Neural Networks.

* [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?utm_source=chatgpt.com)
  Official reference for LSTM implementation and parameters.

---

## 🧾 Word Embeddings & NLP Concepts

* [Word Embeddings Explained (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/?utm_source=chatgpt.com)
  Visual explanation of how embeddings capture semantic relationships between words.

* [Word2Vec Paper (Mikolov et al.)](https://arxiv.org/abs/1301.3781?utm_source=chatgpt.com)
  Foundational paper introducing distributed word representations.

---

## 📊 Sequence Modeling & NLP Theory

* [CS224N Stanford Lecture Notes (NLP with Deep Learning)](https://web.stanford.edu/class/cs224n/?utm_source=chatgpt.com)
  Academic lecture series covering RNNs, LSTMs, and modern NLP architectures.

* [Sequence Modeling Overview (Towards Data Science)](https://towardsdatascience.com/sequence-models-and-long-short-term-memory-networks-3b7b1535bfc3?utm_source=chatgpt.com)
  Practical explanation of sequence modeling concepts in deep learning.

---

## 📦 Datasets & Tools

* [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/?utm_source=chatgpt.com)
  Official documentation for loading and processing NLP datasets like IMDB.

* [IMDB Dataset (Stanford AI)](https://ai.stanford.edu/~amaas/data/sentiment/?utm_source=chatgpt.com)
  Original IMDB sentiment dataset used for binary classification tasks.

---

## ⚙️ Frameworks

* [PyTorch Official Website](https://pytorch.org/?utm_source=chatgpt.com)
  Deep learning framework used to implement RNN and LSTM models.
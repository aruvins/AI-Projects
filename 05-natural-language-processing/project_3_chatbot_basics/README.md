# 🟡 Chatbot Basics

A beginner NLP project that demonstrates how conversational AI systems work.

---

## Concepts Covered

- Tokenization
- Stemming
- Bag-of-Words
- Intent Recognition
- Neural Networks
- Cross Entropy Loss
- Response Generation

---

## Project Pipeline

```text
User Input
      ↓
Tokenization
      ↓
Stemming
      ↓
Bag-of-Words
      ↓
Neural Network
      ↓
Intent Prediction
      ↓
Response Selection
```

---

# 📁 Project Structure

```text
project_3_chatbot_basics/

│
├── data/
│   └── intents.json            # Labeled conversational dataset
│
├── utils/
│   ├── model.py                # Neural network architecture
│   └── bag_of_words.py         # NLP preprocessing utilities  
│
├── train.py                    # Model training pipeline
├── chatbot.py                  # Inference / chat interface
├── main.py                     # Entry point (train + run)
│
├── chatbot_model.pth           # Saved trained model (generated)
│
├── requirements.txt
└── README.md
```



---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 05-natural-language-processing/project_3_chatbot_basics
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

## Example

```text
You: hello
Bot: Hi there!

You: thanks
Bot: You're welcome!

You: bye
Bot: Goodbye!
```

---

## AI Concepts Learned

### NLP Preprocessing

Convert text into machine-readable features.

### Intent Classification

Predict the user's goal.

### Neural Networks

Learn relationships between words and intents.

### Conversational AI

Map user intent to an appropriate response.

This project serves as a foundation for more advanced chatbots using RNNs, LSTMs, Transformers, and Large Language Models.


---

# 🧠 High-Level System Architecture

```text
                ┌──────────────────────┐
                │    User Input        │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │  Tokenization        │
                │  ("hello there")     │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │  Stemming           │
                │  ("hello", "there") │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │ Bag-of-Words Vector  │
                │ [0,1,0,0,1,...]      │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │ Neural Network       │
                │ (MLP Classifier)     │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │ Intent Prediction    │
                │ (greeting/help/etc.) │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │ Response Selection   │
                │ (random template)    │
                └──────────────────────┘
```

---

# 📊 Core AI Concepts (Deep Explanation)

---

## 1. Tokenization (Text → Words)

### What it does

Splits raw text into meaningful units (tokens).

### Example

```text
Input:
"hello how are you"

Output:
["hello", "how", "are", "you"]
```

### Why it matters

Neural networks cannot understand raw text — they require structured inputs.

Tokenization is the **first step in converting language → data**.

---

## 2. Stemming (Normalization)

### What it does

Reduces words to their root form.

```text
"running" → "run"
"played"  → "play"
```

### Why it matters

It reduces vocabulary size and improves generalization:

* "run", "running", "runs" → treated as same feature
* reduces sparsity in ML models

---

# 🌱 Stemming (Deeper Explanation + Mechanism)

Stemming is the process of **reducing a word to its root form by stripping suffixes (and sometimes prefixes)**.

But importantly:

> Stemming does NOT understand meaning — it uses heuristic rules to chop words.

This is what makes it different from “true linguistic understanding”.

---

# 🧠 Core Idea Behind Stemming

Instead of trying to understand language, stemming assumes:

> Words that share a root likely represent the same concept.

So:

```text id="x1"
running → run  
runner   → run  
runs     → run
```

All collapse into a single representation.

---

# ⚙️ How Stemming Actually Works (Mechanism)

Most stemmers (like Porter Stemmer) use **rule-based transformations**.

They do NOT use machine learning.

They apply:

* pattern matching
* suffix stripping rules
* iterative reduction steps

---

# 🔧 Example: Rule-Based Suffix Removal

A simplified idea:

```text id="x2"
IF word ends with "ing"
    remove "ing"
```

So:

```text id="x3"
running → runn → run
```

Another rule:

```text id="x4"
IF word ends with "ed"
    remove "ed"
```

```text id="x5"
played → play
```

---

# 🧠 Porter Stemmer (Real Algorithm Insight)

The most famous stemmer is the **Porter Stemmer (1980)**.

It uses **a sequence of rule “stages”**, not just one-pass stripping.

---

## 🏗 It works in phases:

### Step 1: Plural reduction

```text id="x6"
caresses → caress
ponies   → poni
```

---

### Step 2: Verb reduction

```text id="x7"
relational → relate
conditional → condition
```

---

### Step 3: Suffix trimming

```text id="x8"
revival → reviv
```

---

### Step 4: Final cleanup

```text id="x9"
probate → probat → prob
```

---

# 🧠 Key Insight

Stemming is:

> A deterministic rule engine over character patterns

NOT:

* grammar understanding
* semantic reasoning
* context awareness

---

# ⚡ Why It Works (Even Though It's “Dumb”)

Even though stemming is crude, it helps ML systems because:

---

## 1. Reduces Vocabulary Size

Instead of:

```text id="x10"
run, running, runs, runner
```

you get:

```text id="x11"
run
```

This dramatically reduces:

* input dimensionality
* sparsity in BoW vectors
* training complexity

---

## 2. Improves Statistical Learning

ML models (like your chatbot MLP) don’t understand meaning — they learn correlations.

So:

```text id="x12"
"running" and "run" → same feature
```

means:

* more training samples per feature
* stronger weight updates
* better generalization

---

## 3. Acts as a “Weak Semantic Compression”

Even though it's not semantic, it *approximate-collapses* related words.

---

# ⚖️ Stemming vs Lemmatization (Important Distinction)

This is a key conceptual upgrade.

| Method        | How it works         | Intelligence level |
| ------------- | -------------------- | ------------------ |
| Stemming      | Rule-based chopping  | Low                |
| Lemmatization | Dictionary + grammar | High               |

---

## Example comparison:

### Stemming:

```text id="x13"
better → better (or "bet")
```

### Lemmatization:

```text id="x14"
better → good
```

---

# 🧠 Why Stemming Fails (Important Limitation)

Because it ignores meaning:

---

## ❌ Over-stemming

```text id="x15"
university → univers
```

Now unrelated words may collide:

```text id="x16"
universe → univers
```

Different meanings, same stem → bad feature noise.

---

## ❌ Under-stemming

```text id="x17"
running → run
runner → runner   (not reduced)
```

Now similar words remain separate → redundancy.

---

# 🧠 Intuition: What Stemming Really Is

A better mental model:

> Stemming compresses text by approximating word morphology using heuristics, not linguistics.

It is:

```text id="x18"
lossy compression of language structure
```

---

# 🔗 Why This Matters in Your Chatbot Project

In your pipeline:

```text id="x19"
Tokenization → Stemming → BoW → Neural Network
```

Stemming is doing something very important:

### It makes the feature space more learnable

Without stemming:

* BoW vector becomes extremely sparse
* vocabulary explodes
* model overfits on rare word forms

With stemming:

* features become denser
* patterns become clearer
* intent classification becomes easier

---

# 🧠 Connection to Modern NLP (VERY IMPORTANT)

Modern Transformers like BERT:

> do NOT use stemming at all

Instead they use:

* subword tokenization (WordPiece / BPE)

Example:

```text id="x20"
running → run + ##ning
```

So instead of deleting structure, they **preserve it in fragments**.

---

# 🚀 Key Takeaway (High-Level Insight)

Stemming represents an early stage in NLP where:

> We forced language into simplified symbolic rules to make it usable for machine learning.

Modern NLP replaces this with:

> learned representations that preserve full linguistic structure.

---

## 3. Bag-of-Words (BoW)

### What it is

A **numerical representation of text** where each word becomes a feature.

### Example

Vocabulary:

```text
["hello", "bye", "thanks"]
```

Sentence:

```text
"hello thanks"
```

Vector:

```text
[1, 0, 1]
```

---

### Key Idea

> We ignore grammar and word order — only presence matters.

---

### Limitations

* no context
* no semantics
* no word relationships

This is why modern models use embeddings instead.

---

## 4. Intent Classification (Supervised Learning)

### Problem Type

This is a:

> Multi-class classification problem

### Input → Output mapping

```text
"hello"        → greeting
"goodbye"      → farewell
"thank you"    → gratitude
```

---

### Model learns:

```text
P(intent | sentence)
```

Instead of generating text, it predicts a label.

---

# 🎯 Intent Classification (Deeper Explanation)

Intent classification is a **supervised learning problem** where the goal is:

> Map a user’s sentence → a predefined category (intent)

So instead of generating language, the model is doing:

```text id="i1"
f(text) → intent label
```

---

# 🧠 What “Intent” Actually Means

An intent is simply:

> the underlying goal of a user message

Examples:

| User Input        | Intent     |
| ----------------- | ---------- |
| "hello"           | greeting   |
| "what time is it" | time_query |
| "thanks"          | gratitude  |
| "bye"             | farewell   |

So the system is not answering words — it is classifying **human purpose**.

---

# 🏗️ Formal ML View

Intent classification is:

> A multiclass classification problem over text features

Mathematically:

P(y \mid x)

Where:

* ( x ) = input sentence
* ( y ) = intent class

The model outputs:

```text id="i2"
P(greeting | "hello") = 0.92
P(farewell | "hello") = 0.03
P(gratitude | "hello") = 0.05
```

Then we take:

```text id="i3"
argmax(P(y|x))
```

---

# ⚙️ How the Mechanism Works (Step-by-Step)

Your pipeline looks like this:

```text id="i4"
Text → BoW → Neural Network → Softmax → Intent
```

Let’s break it down.

---

# 1. Text → Feature Vector (BoW Layer)

The sentence is converted into a vector:

Example vocabulary:

```text id="i5"
["hello", "bye", "thanks"]
```

Input:

```text id="i6"
"hello thanks"
```

Becomes:

```text id="i7"
[1, 0, 1]
```

So the model no longer sees words — it sees:

> presence patterns in a high-dimensional space

---

# 2. Neural Network = Pattern Recognizer

Now the vector goes into a Multi-Layer Perceptron (MLP):

```text id="i8"
x → W1 → ReLU → W2 → ReLU → W3
```

Each layer transforms representation:

### Layer 1:

Detects simple patterns:

* “hello present”
* “thanks present”

### Layer 2:

Combines patterns:

* “hello + thanks → greeting + gratitude mix”

### Layer 3:

Maps to intent space:

* greeting
* farewell
* gratitude

---

# 🧠 Key Idea

The network is learning:

> which combinations of words correspond to which user intent

Not meaning — just statistical structure.

---

# 3. Softmax = Probability Conversion

The final layer outputs raw scores:

```text id="i9"
[2.1, 0.3, -1.2]
```

These are called **logits**.

Softmax converts them into probabilities:

\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}

So outputs become:

```text id="i10"
greeting: 0.85
farewell: 0.10
gratitude: 0.05
```

---

# 🧠 Why Softmax is Important

Softmax ensures:

* outputs sum to 1
* interpretability as probabilities
* sharper decision boundaries

So the model can say:

> “I am 85% sure this is a greeting”

---

# 📉 Training Mechanism (How It Learns)

The model learns using:

## Cross-Entropy Loss

It compares:

```text id="i11"
predicted distribution vs true label
```

If correct intent is:

```text id="i12"
greeting
```

but model predicts:

```text id="i13"
farewell
```

then loss is high.

---

## Learning process:

```text id="i14"
1. Forward pass (prediction)
2. Compute loss
3. Backpropagation
4. Weight updates (gradient descent)
```

Over time:

> word patterns → intent mapping becomes stronger

---

# 🧠 What the Model Is Really Learning

This is the key conceptual insight:

The model is learning a function:

f(x) = \text{MLP}(\text{BoW}(x))

But in reality:

> it is learning statistical associations between word patterns and labels

---

# 🔍 Geometric Interpretation (Very Important)

In vector space:

* each sentence = a point
* each intent = a region

So:

```text id="i15"
greeting sentences cluster together
farewell sentences cluster together
```

The neural network learns **decision boundaries** between these clusters.

---

# ⚖️ Why This Works (Even With Simple Models)

Even simple MLPs work well because:

### 1. Intent is low-dimensional

You don’t need full language understanding to detect:

* greeting
* goodbye
* thanks

These are **surface-level semantic classes**

---

### 2. Word patterns are strong signals

Example:

```text id="i16"
"hello", "hi", "hey"
```

are highly predictive of “greeting”.

So classification is mostly:

> pattern matching + nonlinear weighting

---

# 🚫 Limitations (Important Insight)

Intent classification fails when:

## 1. Context matters

```text id="i17"
"I don’t like saying goodbye"
```

contains “goodbye” but is NOT farewell intent.

---

## 2. Multi-intent sentences

```text id="i18"
"hi, can you help me?"
```

contains:

* greeting
* request

---

## 3. Out-of-distribution inputs

```text id="i19"
"quantum entanglement in finance"
```

model has no intent class → misclassification

---

# 🚀 How This Evolves Into Modern AI

Intent classification is the **ancestor of modern language understanding systems**.

| Stage        | Model                        |
| ------------ | ---------------------------- |
| This project | BoW + MLP                    |
| Next step    | TF-IDF + Logistic Regression |
| Advanced NLP | RNN/LSTM sequence models     |
| Modern NLP   | Transformer encoders (BERT)  |
| Current AI   | LLM-based intent + reasoning |

---

# 🧠 Final Mental Model

Intent classification is:

> converting unstructured language into structured decision categories using statistical pattern learning.

Or even more simply:

```text id="i20"
Language → Features → Probability over intents
```

---

## 5. Neural Network (MLP Classifier)

### Architecture

```text
Input Layer (BoW vector)
        ↓
Hidden Layer (ReLU)
        ↓
Hidden Layer (ReLU)
        ↓
Output Layer (intent probabilities)
```

---

### What it learns

The model learns:

* which words correlate with which intents
* patterns in word combinations
* nonlinear relationships between features

---

### Output Example

```text
greeting: 0.92
goodbye : 0.03
thanks  : 0.05
```

---

## 6. Cross Entropy Loss

### Purpose

Measures how wrong the model is.

---

### Intuition

If correct label is:

```text
greeting
```

but model predicts:

```text
goodbye (high probability)
```

→ loss is high

---

### Goal of training

Minimize:

```text
Loss → 0
```

---

## 7. Gradient Descent (Learning Mechanism)

The model improves using:

* forward pass
* loss calculation
* backpropagation
* weight updates

This is how it “learns language patterns”.

---

## 8. Response Generation (Rule-Based Layer)

This project does NOT generate text.

Instead it uses:

```text
Intent → Predefined Responses
```

Example:

```json
"greeting": [
  "Hello!",
  "Hi there!"
]
```

Then randomly selects one.

---

### Why this matters

This is the **hybrid chatbot architecture**:

```text
ML (understanding)
+ Rule-based system (generation)
```

---

# 🔁 Full ML Pipeline Summary

```text
Raw Text Input
      ↓
Tokenization
      ↓
Stemming
      ↓
Vectorization (BoW)
      ↓
Neural Network Classifier
      ↓
Intent Label Prediction
      ↓
Response Mapping
      ↓
Chatbot Output
```

---

# 🚀 Why This Project Matters

This project teaches the **core foundation of NLP systems before transformers existed**.

You are learning:

### Classical NLP Stack

* feature engineering (BoW)
* supervised learning
* classification models

### Machine Learning Stack

* PyTorch neural networks
* loss functions
* optimization

### System Design Stack

* pipeline architecture
* inference vs training separation
* model serialization

---

# 🔥 Limitations (Important for AI progression)

This chatbot:

* ❌ does not understand context
* ❌ cannot remember conversation history
* ❌ cannot generate new sentences
* ❌ ignores word order
* ❌ cannot handle unseen intents well

---

# 🚀 How This Evolves Into Modern AI

This project is the **entry point** into modern NLP:

| Stage        | Model Type                        |
| ------------ | --------------------------------- |
| This project | Bag-of-Words + MLP                |
| Next step    | TF-IDF + Logistic Regression      |
| Advanced NLP | RNN / LSTM                        |
| Modern NLP   | Transformers (BERT)               |
| Current SOTA | Large Language Models (GPT-style) |


---

# 📚 References

This project is based on foundational concepts in Natural Language Processing and deep learning. The following resources were used to guide the design and implementation.

---

## 📖 NLP & Machine Learning Foundations

* Jurafsky, D., & Martin, J. H.
  **Speech and Language Processing (3rd ed. draft)**
  [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)

* Manning, C. D., Raghavan, P., & Schütze, H.
  **Introduction to Information Retrieval**
  [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

---

## 🧠 Neural Networks & Deep Learning

* Goodfellow, I., Bengio, Y., & Courville, A.
  **Deep Learning (MIT Press, 2016)**
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

* Stanford CS231n
  **Convolutional Neural Networks for Visual Recognition (Neural Net fundamentals also applicable)**
  [https://cs231n.github.io/](https://cs231n.github.io/)

---

## 🤖 Intent Classification & NLP Pipelines

* Rao, D., & McMahan, B.
  **Natural Language Processing for Chatbots** (Google AI insights)

* Raschka, S.
  **Machine Learning with PyTorch and Scikit-Learn**
  [https://sebastianraschka.com/](https://sebastianraschka.com/)

---

## 🧰 Tools & Libraries

* Python NLTK Library
  [https://www.nltk.org/](https://www.nltk.org/)

* PyTorch Deep Learning Framework
  [https://pytorch.org/](https://pytorch.org/)

* NumPy Numerical Computing
  [https://numpy.org/](https://numpy.org/)

---

## 💡 Conceptual Inspiration

* Rule-based chatbot architectures (ELIZA, 1966)
* Bag-of-Words text representation models
* Early intent classification systems used in customer support bots
* Modern transformer-based chatbots (BERT, GPT family)

---

## 🚀 Modern Extensions (for future work)

This project serves as a baseline for:

* Recurrent Neural Networks (RNNs, LSTMs)
* Attention mechanisms
* Transformer architectures (Vaswani et al., 2017)
* Large Language Models (GPT-style systems)

---

## 📄 Key Paper (Transformers)

* Vaswani et al. (2017)
  **Attention Is All You Need**
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

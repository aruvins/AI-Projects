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

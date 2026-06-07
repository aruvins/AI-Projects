# 🟢 Text Summarization System (Classical NLP → Transformers)

This project explores how machines compress long text into short, meaningful summaries using both **classical NLP techniques** and **modern transformer-based models**.

It is designed as a learning progression from:
> **frequency-based text understanding → statistical sentence ranking → deep learning-based language generation**

---

# 🎯 Goal

Build a system that can:
- Read long-form text (articles, news, documents)
- Identify important information
- Generate a short summary

And more importantly:
> Understand *how and why* different AI approaches solve this problem.

---

# 🧠 Core AI Idea: Information Compression

Text summarization is fundamentally a **compression problem**:

Long Document → Compact Representation → Short Summary

There are two ways to do this:

| Approach | Idea | Output Type |
|----------|------|------------|
| Extractive | Select important sentences | Copy from text |
| Abstractive | Generate new sentences | Rewritten summary |

---

# 📦 Project Structure


```text
project_2_text_summarization/
│
├── data/
│   └── article.txt
│
├── models/
│   ├── extractive.py        # classical NLP summarizer
│   └── transformer.py       # deep learning summarizer
│
├── utils/
│   ├── preprocess.py        # tokenization & cleaning
│   └── evaluator.py        # compression metrics
│
├── main.py
└── requirements.txt

```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 05-natural-language-processing/project_2_text_summarization
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

# 🧹 1. Text Preprocessing (Foundation Layer)

Before any AI model can understand text, we must convert raw language into structured tokens.

## Key Steps

### Lowercasing
Reduces vocabulary complexity:

```text
Apple → apple
APPLE → apple
```

### Tokenization
Splits text into words:

```text

"I love AI"
→ ["i", "love", "ai"]

```

### Sentence Segmentation
Splits documents into units for ranking:

```text

Sentence 1
Sentence 2
Sentence 3

```

---

## Why preprocessing matters

Without it:
- Same words are treated as different tokens
- Models learn noise instead of meaning

---

# 🧠 2. Extractive Summarization (Classical NLP)

This is a **statistical approach** to summarization.

## Core Assumption

> Important words appear more frequently in important sentences.

---

## Step 1: Word Frequency Model

We compute how often each word appears:

```text
assistant → 5
apple → 2
productivity → 3
```

This is a simple form of **statistical importance estimation**.

---

## Step 2: Sentence Scoring

Each sentence is scored based on word importance:

```python
Sentence score = sum(word frequencies)
```

So:

```text
"Apple announced a new AI assistant"
→ high score
```

---

## Step 3: Sentence Selection

We select top-k scoring sentences:

```text
Top sentences → final summary
```

---

## AI Concept Insight

This method assumes:

```text
frequency ≈ importance
```

Limitations:
- ignores word order
- ignores meaning
- cannot paraphrase

But it is:
 
✔ fast  
✔ interpretable  
✔ strong baseline  

---

# 📊 3. TF-IDF (Improved Statistical Understanding)

TF-IDF improves simple frequency counting.

## Intuition

Not all words are equally important:

| Word | Importance |
|------|------------|
| the | low |
| is | low |
| AI | high |
| productivity | high |

---

## Formula

```math
TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)
```

Where:

- TF = term frequency in document

```math
TF(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in document } d}
```

- IDF = how rare the word is across all documents

```math
IDF(t, D) = \log \left( \frac{N}{1 + |\{ d \in D : t \in d \}|} \right)
```

And:

* ( t ) = term (word)
* ( d ) = document
* ( D ) = corpus (all documents)
* ( N ) = total number of documents



---

## Why TF-IDF works better

It suppresses:
- common words (“the”, “is”)

And highlights:
- rare informative words (“inflation”, “merger”, “AI”)

---

# 🤖 4. Transformer-Based Summarization (Modern AI)

This project also includes a deep learning model using pretrained transformers.

Example model:
- :contentReference[oaicite:1]{index=1}

---

## Core Idea

Instead of counting words, transformers learn:

```text

Meaning + Context + Relationships

```

---

## Architecture

```text

Input Text
↓
Tokenizer
↓
Encoder (understanding)
↓
Decoder (generation)
↓
Summary

```

---

## Key Mechanism: Attention

Attention allows the model to focus on important words:

Example:

```text

Apple announced a new AI assistant

```

The model focuses on:
- Apple
- AI assistant

and ignores:
- irrelevant details

---

## Why Transformers outperform classical NLP

| Feature | Classical NLP | Transformers |
|--------|---------------|-------------|
| Understand meaning | ❌ | ✔ |
| Generate new text | ❌ | ✔ |
| Context awareness | Limited | Strong |
| Paraphrasing | ❌ | ✔ |

---

# 🔁 5. Extractive vs Abstractive Comparison

| Feature | Extractive | Abstractive |
|----------|------------|-------------|
| Output | Original sentences | New sentences |
| Flexibility | Low | High |
| Accuracy | High factual fidelity | May hallucinate |
| Complexity | Low | High |

---

# 📉 6. Evaluation Metric

We measure compression using:

```text

compression ratio = summary length / original length

```

Lower ratio → more compression.

---

# ⚙️ 7. Pipeline Overview

## Extractive Pipeline


Raw Text
→ Preprocessing
→ Sentence Splitting
→ Word Frequency
→ Sentence Ranking
→ Summary


## Transformer Pipeline

Raw Text
→ Tokenization
→ Encoder
→ Attention Layers
→ Decoder
→ Summary


---

# 🧪 8. Key AI Concepts Learned

### NLP Fundamentals
- Tokenization
- Sentence segmentation
- Vocabulary construction

### Statistical NLP
- Word frequency modeling
- TF-IDF weighting
- Sentence scoring

### Deep Learning NLP
- Encoder-decoder architecture
- Self-attention mechanism
- Sequence-to-sequence learning

### Generative AI
- Text generation vs extraction
- Abstraction and paraphrasing
- Pretrained language models

---

# 🚀 9. Why This Project Matters

This project demonstrates the evolution of NLP:


Rule-based systems
↓
Statistical NLP
↓
Deep learning
↓
Transformer-based generative AI

It provides a bridge between:
- classical machine learning
- modern LLM systems like GPT-style models

---

# 📌 10. Future Improvements

- ROUGE score evaluation
- BERT-based extractive ranking
- Fine-tuned summarization model
- Streamlit UI dashboard
- Multi-document summarization
- Real-time news summarizer

---

# 🧠 Final Insight

Summarization is not just a task — it is a study of:

> how machines decide what information is important

This project builds intuition for both:
- classical NLP systems
- modern large language models

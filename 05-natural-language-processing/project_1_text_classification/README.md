# 🧠 IMDB Sentiment Classification (Bag-of-Words vs TF-IDF)

A machine learning project that performs **binary sentiment classification** on the IMDB movie reviews dataset using classical NLP techniques:

* Bag-of-Words (from scratch)
* TF-IDF (scikit-learn)
* Logistic Regression classifier

This project demonstrates how raw text is converted into numerical features for machine learning models.

---

## 🚀 Project Goals

* Learn how text becomes numerical data
* Implement Bag-of-Words manually
* Compare BoW vs TF-IDF performance
* Train a supervised sentiment classifier
* Build a full end-to-end NLP pipeline

---

## 📊 Dataset

We use the official IMDB dataset:

* 25,000 labeled training reviews
* 25,000 labeled test reviews
* Binary labels:

  * `1` → Positive review
  * `0` → Negative review

Dataset source:

* [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## 🧱 Project Structure

```text
project_1_text_classification/

├── main.py
├── models/
│   └── logistic_regression.py
│
├── utils/
│   ├── imdb_loader.py
│   ├── vectorizers.py
│   └── metrics.py
│
├── data/
├── outputs/
└── requirements.txt
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 05-natural-language-processing/project_1_text_classification

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

## 🔀 Choose Feature Type

Inside `main.py`, toggle between:

### TF-IDF (recommended)

```python
USE_TFIDF = True
```

### Bag-of-Words (from scratch)

```python
USE_TFIDF = False
```

---

# 🧠 AI Concepts

---

## 1. 🧹 Text Preprocessing & Tokenization

Before a machine learning model can understand text, raw language must be converted into structured units.

### What happens:

```text
"I loved this movie"
→ ["i", "loved", "this", "movie"]
```

This step is called **tokenization**.

### Why it matters:

Machine learning models cannot process raw strings. They require:

* consistent units (tokens)
* reduced noise (punctuation, casing, etc.)
* normalized representations

### Key design choices in this project:

* Lowercasing text → reduces vocabulary size
* Simple whitespace/token splitting → fast baseline approach
* No stemming/lemmatization → keeps pipeline lightweight and interpretable

### Limitation:

This naive tokenizer does NOT handle:

* negation well ("not good")
* contractions ("didn't")
* punctuation semantics ("!!!")

More advanced systems use:

* subword tokenization (BPE, WordPiece)
* spaCy / NLTK pipelines

---

## 2. 📦 Bag-of-Words Representation

Bag-of-Words (BoW) converts text into a **frequency vector**.

### Core idea:

A document is represented as:

> “How many times each word appears”

### Example:

```text
Text: "the movie was great great"
```

Becomes:

| Word  | Count |
| ----- | ----- |
| the   | 1     |
| movie | 1     |
| was   | 1     |
| great | 2     |

Vector form:

```text
[1, 1, 1, 2]
```

---

### Mathematical formulation:

Let:

* ( d ) = document
* ( w_i ) = word in vocabulary

Then:

```math
X_{d,i} = count(w_i in d)
```

---

### Strengths:

* Simple
* Interpretable
* Works well with linear models
* Fast to compute

---

### Weaknesses:

* Loses word order (no syntax)
* Cannot understand meaning
* Treats all words equally
* Sparse high-dimensional vectors

---

### Key insight:

Bag-of-Words assumes:

> “Word order does not matter, only frequency does.”

This is a **strong but useful approximation**.

---

## 3. 📊 TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF improves Bag-of-Words by weighting words based on **importance**.

---

### 3.1 Term Frequency (TF)

Measures how often a word appears in a document:

```text
TF(w, d) = count(w in d)
```

---

### 3.2 Inverse Document Frequency (IDF)

Measures how rare a word is across all documents:

```text
IDF(w) = log(N / DF(w))
```

Where:

* ( N ) = total number of documents
* ( DF(w) ) = number of documents containing word ( w )

---

### 3.3 Final TF-IDF score:

```text
TF-IDF(w, d) = TF(w, d) × IDF(w)
```

---

### Intuition:

| Word        | Effect                                  |
| ----------- | --------------------------------------- |
| "movie"     | common → downweighted                   |
| "excellent" | rare → upweighted                       |
| "the"       | extremely common → heavily downweighted |

---

### Why TF-IDF works better:

It emphasizes **discriminative words**, which are more useful for classification.

Example:

```text
"I thought the movie was absolutely amazing"
```

Important words:

* amazing ↑
* absolutely ↑

Ignored words:

* the ↓
* was ↓

---

### Strengths:

* Handles common words better than BoW
* Strong baseline for text classification
* Works extremely well with linear models

---

### Weaknesses:

* Still ignores word order
* Cannot model semantics
* Cannot capture context ("not good")

---

## 4. 🤖 Logistic Regression for Text Classification

Despite its name, logistic regression is a **linear classification model**, not a regression model.

---

### Core idea:

It learns a linear decision boundary:

```text
y = sigmoid(w · x + b)
```

Where:

* ( x ) = text vector (BoW or TF-IDF)
* ( w ) = learned weights
* ( b ) = bias term

---

### Sigmoid function:

```text
σ(z) = 1 / (1 + e^{-z})
```

Outputs probability:

```text
P(y = 1 | x)
```

---

### Decision rule:

```text
if P > 0.5 → Positive
else → Negative
```

---

### Why logistic regression works well for NLP:

* High-dimensional sparse data
* Linear separability often sufficient
* Robust to noise
* Efficient on large datasets (IMDB = 50k samples)

---

### What the model learns:

Each word gets a weight:

| Word     | Weight |
| -------- | ------ |
| amazing  | +2.4   |
| great    | +1.8   |
| boring   | -2.1   |
| terrible | -2.8   |

So prediction becomes:

> "sum of sentiment signals across words"

---

## 5. 📉 High-Dimensional Sparse Representations

Text vectors are:

* Very large (10,000–100,000 dimensions)
* Mostly zeros

Example:

```text
[0, 0, 0, 1, 0, 0, 3, 0, ...]
```

---

### Why sparsity matters:

Efficient ML algorithms like logistic regression:

* exploit sparsity
* avoid storing full matrices
* scale to large vocabularies

---

### Key concept:

This is called the:

> “curse of dimensionality”

but linear models handle it surprisingly well.

---

## 6. ⚖️ Bias–Variance Tradeoff in This Project

### Bag-of-Words:

* High variance (overfits noise more easily)
* Lower bias

### TF-IDF:

* Lower variance
* Better generalization

---

### Why TF-IDF usually wins:

It reduces noise from:

* stopwords
* frequent irrelevant terms
* dataset-specific artifacts

---

## 7. 📊 Evaluation Metrics

### Accuracy:

```text
(correct predictions) / (total predictions)
```

---

### Precision / Recall:

* Precision → correctness of positive predictions
* Recall → coverage of actual positives

---

### F1 Score:

Harmonic mean of precision and recall:

```text
F1 = 2 * (P × R) / (P + R)
```

---

### Confusion Matrix:

|          | Predicted + | Predicted - |
| -------- | ----------- | ----------- |
| Actual + | TP          | FN          |
| Actual - | FP          | TN          |

---

## 8. 🧠 Why This Pipeline Works

This system works because:

1. Language contains strong statistical signals
2. Sentiment words carry high predictive power
3. Linear models can separate sentiment space well
4. TF-IDF highlights meaningful tokens


---


## 📈 Results

Typical performance:

| Method       | Accuracy |
| ------------ | -------- |
| Bag-of-Words | ~84–87%  |
| TF-IDF       | ~88–90%  |

---

## 📉 Output Example

```
Accuracy: 0.8892

Classification Report:
precision    recall  f1-score
```

Confusion matrix is saved to:

```text
outputs/confusion_matrix.png
```

---

## 💬 Interactive Prediction

After training:

```text
Enter review (or 'quit'):
```

Example:

```text
I really loved this movie, it was amazing
Prediction: Positive
```

```text
This was boring and too long
Prediction: Negative
```

---

## 🧪 Key Learnings

This project teaches:

* How text becomes vectors
* Difference between BoW vs TF-IDF
* Why feature engineering matters
* Basics of supervised NLP pipelines
* How logistic regression works on high-dimensional text

---

## 📌 Future Improvements

You can extend this project with:

* 🔥 N-grams (bigrams, trigrams)
* 📊 Cross-validation tuning
* 🤖 Word embeddings (Word2Vec / GloVe)
* 🧠 RNN / LSTM models
* ⚡ Transformer-based classifiers (BERT)

---

## 📚 References

* IMDB Dataset: [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
* scikit-learn TF-IDF: [https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* Logistic Regression: [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

### Core NLP & Feature Engineering

* Jurafsky & Martin — *Speech and Language Processing*
  [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)

* Manning, Raghavan, Schütze — *Introduction to Information Retrieval*
  [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

---

### TF-IDF

* scikit-learn TF-IDF documentation
  [https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

* Sparck Jones (1972) — Original IDF concept
  [https://dl.acm.org/doi/10.1145/361219.361220](https://dl.acm.org/doi/10.1145/361219.361220)

---

### Logistic Regression

* Hastie, Tibshirani, Friedman — *Elements of Statistical Learning*
  [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)

* scikit-learn Logistic Regression
  [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

### IMDB Dataset

* Maas et al. (2011) — Learning Word Vectors for Sentiment Analysis
  [https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

---

## 🏁 Summary

This project is a foundational NLP pipeline that shows how classical machine learning approaches can still achieve strong performance on sentiment analysis tasks before deep learning methods are introduced.

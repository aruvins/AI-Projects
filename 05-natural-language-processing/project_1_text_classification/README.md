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

## 🧠 How It Works

### 1. Text Preprocessing

```text
"I loved this movie"
→ ["i", "loved", "this", "movie"]
```

---

### 2. Feature Extraction

#### Bag-of-Words

Counts word frequency:

```text
great → 3
movie → 1
```

Vector form:

```text
[1, 1, 3, 0, ...]
```

---

#### TF-IDF

Weights words by importance:

* Reduces impact of common words
* Increases importance of meaningful words

---

### 3. Model

We use:

```text
Logistic Regression
```

to learn a boundary between:

```text
negative reviews ← | → positive reviews
```

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

---

## 🏁 Summary

This project is a foundational NLP pipeline that shows how classical machine learning approaches can still achieve strong performance on sentiment analysis tasks before deep learning methods are introduced.

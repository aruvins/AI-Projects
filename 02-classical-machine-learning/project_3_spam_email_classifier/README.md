# 🟢 Spam Email Classifier (Project 3)

A beginner-friendly machine learning project that classifies emails (or SMS messages) as **Spam** or **Not Spam (Ham)** using classical NLP and machine learning models.

This project helps you understand how text becomes data and how models learn patterns in language.

---

## 📁 Project Structure

```
project_3_spam_email_classifier/
│
├── data/
│   └── spam.csv                  # dataset (auto-generated)
│
├── output/
│   ├── naive_bayes/
│   └── logistic_regression/
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       └── sample_predictions.txt
│
├── src/
│   └── download_data.py         # dataset downloader
│
├── main.py                      # training + evaluation script
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 02-classical-machine-learning/project_3_spam_email_classifier
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

# 4. Run The Project

```bash
python main.py
```

---

# 🎯 Goal

Learn how to:

* Turn raw text into numerical data
* Train machine learning models for classification
* Evaluate model performance using metrics
* Compare multiple models

---

# 🧠 Key Concepts (Beginner Friendly)

## 1. 📦 Bag of Words (BoW)

Machine learning models cannot understand text directly.

So we convert text into numbers by counting words.

Example:

```
"Win! win free prize. Win now!"
```

becomes:

```
win → 3
free → 1
prize → 1
now → 1
```

This ignores grammar and order, only focuses on word presence.

---

## 2. 📊 TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF improves Bag of Words by weighting words:

TF-IDF determines “importance” using a simple idea:

> A word is important if it appears a lot in one message, but not in many messages overall.

It measures this using two scores:

---

## 📦 1. Term Frequency (TF)

> How often a word appears in a single message

If a word shows up more times in one email, its TF goes up.

Example:

* “win win win” → high TF for “win”
* “free” once → lower TF

So TF answers:

> “How important is this word in THIS message?”

---

## 🌍 2. Inverse Document Frequency (IDF)

> How rare the word is across all messages

If a word appears in almost every email, it gets a low score.

If it appears in only a few emails, it gets a high score.

So IDF answers:

> “How useful is this word for telling messages apart?”

---

## ⚖️ Final importance score

TF-IDF multiplies them:

TF-IDF = TF * IDF

---

## 🧠 Intuition (simple explanation)

* Common words like “the”, “and” → low importance
* Spam words like “win”, “free”, “urgent” → high importance

Because:

* They appear often in spam messages (high TF)
* But not in normal messages (high IDF)


---

## 3. 🧹 NLP Preprocessing

Before training, text is cleaned:

* Lowercasing
* Removing stop words (“and”, “the”, “is”)
* Tokenizing into words

This helps models focus on meaningful patterns.

---

# 🤖 1. Naive Bayes (Multinomial Naive Bayes)

## 🧠 Core idea

Naive Bayes is based on **probability and word frequency**.

It asks:

> “Given the words in this message, what is the probability it is spam vs not spam?”

---

## 📦 How it represents text

After TF-IDF or Bag-of-Words, a message becomes:

```
"win free prize"
```

→ vector like:

```
[win: 1, free: 1, prize: 1, other_words: 0]
```

---

## 📊 The math intuition (simple version)

Naive Bayes computes:

```math
P(Spam \mid Words) = \frac{P(Words \mid Spam) \cdot P(Spam)}{P(Words)}
```

We don’t actually compute all of that directly in code — we compare:

* Probability message is spam
* Probability message is not spam

and choose the bigger one.

---

## 🧩 The “Naive” assumption

It assumes:

> Each word is independent of the others

So:

* “free” doesn’t depend on “win”
* “urgent” doesn’t depend on “account”

This is NOT true in real language — but surprisingly works very well.

---

## 📈 Why it works so well for spam

Spam messages have very strong word signals:

| Spam words | Effect                |
| ---------- | --------------------- |
| free       | strong spam indicator |
| win        | strong spam indicator |
| urgent     | strong spam indicator |

Naive Bayes just counts these patterns.

---

## ⚡ Strengths

* Extremely fast
* Works well with small datasets
* Very strong baseline for text classification
* Handles high-dimensional sparse data (like TF-IDF) very well

---

## ❌ Weaknesses

* Assumes word independence (not realistic)
* Can miss complex language patterns
* Not great for context-heavy tasks

---

# 📊 2. Logistic Regression

## 🧠 Core idea

Logistic Regression learns:

> “Which words push a message toward spam or ham?”

Instead of probabilities from counts, it learns **weights for each word**.

---

## 📦 How it works with text

Each word gets a weight:

| Word    | Weight                |
| ------- | --------------------- |
| free    | +2.3 (spam indicator) |
| win     | +2.0 (spam indicator) |
| meeting | -1.8 (ham indicator)  |

---

## 🧮 The math intuition

It computes a score:

```math
z = w \cdot x + b
```

Then converts it into a probability:

```math
P(y=1) = \frac{1}{1 + e^{-z}}
```

Where:

* `z` = weighted sum of word importance
* `y=1` = spam

---

## 🧠 What it is really doing

It is learning a **boundary** between spam and ham messages in a high-dimensional space.

Imagine:

* spam messages cluster in one region
* ham messages cluster in another

Logistic Regression draws a line (or hyperplane) between them.

---

## 📈 Why it works well for spam

Spam detection is basically:

> “Certain words strongly push classification in one direction”

Logistic Regression is perfect for that because:

* it learns **importance of each word**
* it combines words instead of treating them independently
* it handles overlapping signals better than Naive Bayes

---

## ⚡ Strengths

* More accurate than Naive Bayes in many cases
* Interpretable (you can inspect word weights)
* Works very well with TF-IDF
* Stable and widely used in industry

---

## ❌ Weaknesses

* Slightly slower than Naive Bayes
* Can struggle with very complex nonlinear patterns
* Still not as powerful as deep learning models

---

# ⚖️ Naive Bayes vs Logistic Regression (simple comparison)

| Feature          | Naive Bayes               | Logistic Regression   |
| ---------------- | ------------------------- | --------------------- |
| Speed            | 🚀 Very fast              | 🚀 Fast               |
| Accuracy         | Good baseline             | Usually better        |
| Interpretability | Medium                    | High                  |
| Assumptions      | Strong (independence)     | Fewer assumptions     |
| Best for         | quick text classification | real-world ML systems |

---

# 🧠 Why we used THESE two models in your project

This project is intentionally designed as a **classical NLP pipeline**, so these models are ideal because:

---

## 1. They work extremely well with TF-IDF

TF-IDF produces **high-dimensional sparse vectors**, and both models are built for that.

---

## 2. They are industry-standard baselines

Before using deep learning models like BERT, people ALWAYS start with:

* Naive Bayes
* Logistic Regression

Because they give a strong reference point.

---

## 3. They are fast and lightweight

You can train them in milliseconds to seconds — perfect for learning and iteration.

---

## 4. They are interpretable

You can actually understand:

* which words caused spam prediction
* how the model is making decisions

This is very important for learning ML.

---

# 🚀 Big picture intuition

Your full pipeline is:

```text
Email text
   ↓
TF-IDF (turn words into numbers)
   ↓
Model (Naive Bayes or Logistic Regression)
   ↓
Spam / Ham prediction
```

Both models are basically learning:

> “Which words are signals of spam vs normal communication?”

---

## 5. 📏 Evaluation Metrics

We measure performance using:

* **Accuracy** → overall correctness
* **Precision** → how many predicted spam were actually spam
* **Recall** → how many actual spam were found
* **Confusion Matrix** → breakdown of predictions

---

## 3. What happens when you run it

### Step 1: Dataset download

Automatically downloads the SMS Spam dataset (or uses fallback synthetic data).

---

### Step 2: Training

Two models are trained:

* Naive Bayes
* Logistic Regression

---

### Step 3: Evaluation

For each model, the program:

* Prints accuracy
* Prints classification report
* Generates confusion matrix

---

### Step 4: Visual outputs

Saved in:

```
output/
```

Each model gets:

* 📊 confusion matrix image
* 📄 classification report
* 🧪 sample predictions

---

# 📊 Example Output

## Accuracy

```
Accuracy: 0.98
```

---

## Confusion Matrix

```
            Predicted
            Ham   Spam
Actual Ham   940    5
       Spam   12   158
```

---

## Sample Predictions

```
SPAM: WIN a FREE iPhone now!!!
HAM: Hey, are we still meeting today?
SPAM: Urgent! Your account has been suspended
```

---

# 🧪 Why This Project Matters

This project teaches the foundation of:

* NLP (Natural Language Processing)
* Spam detection systems (used in Gmail, Outlook)
* Real-world text classification pipelines

---

# 🚀 Possible Improvements

If you want to level this up:

### 🔥 Model upgrades

* Support Vector Machines (SVM)
* XGBoost
* BERT (deep learning NLP)

### 📊 Visual upgrades

* ROC curves
* Word frequency charts
* Interactive dashboards

### 🌐 Deployment

* Flask API
* Streamlit web app
* Real-time email classifier

---

# 🧠 Summary

You built a full machine learning pipeline:

```
Raw Text → TF-IDF → Model → Prediction → Evaluation → Visualization
```

This is the same core pipeline used in real production spam filters.


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
"win free prize"
```

becomes:

```
win → 1
free → 1
prize → 1
```

This ignores grammar and order, only focuses on word presence.

---

## 2. 📊 TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF improves Bag of Words by weighting words:

* Words that appear often in one message → important
* Words that appear everywhere (“the”, “and”) → less important

So spam words like:

* “free”
* “win”
* “urgent”

get higher importance.

---

## 3. 🧹 NLP Preprocessing

Before training, text is cleaned:

* Lowercasing
* Removing stop words (“and”, “the”, “is”)
* Tokenizing into words

This helps models focus on meaningful patterns.

---

## 4. 🤖 Models Used

### 🔹 Naive Bayes

* Works very well for text
* Assumes words are independent
* Fast and strong baseline

### 🔹 Logistic Regression

* Learns weighted importance of words
* Often more accurate than Naive Bayes
* Standard model in NLP classification

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


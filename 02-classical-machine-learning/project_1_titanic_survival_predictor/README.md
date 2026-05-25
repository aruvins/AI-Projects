# 🚢 Titanic Survival Predictor

A beginner-friendly end-to-end Machine Learning project using the famous Titanic dataset.

This project teaches the complete classical machine learning workflow:

```text
raw data
   ↓
preprocessing
   ↓
feature engineering
   ↓
model training
   ↓
evaluation
   ↓
prediction
```

The goal is not just to predict Titanic survival.

The real goal is to understand:
- how machine learning models learn patterns
- how data must be cleaned before training
- how classification systems work
- how models are evaluated

---

# 📚 What You Will Learn

This project introduces the foundations of supervised machine learning.

By completing this project, you will learn:

- how datasets are loaded and analyzed
- how missing data is handled
- how categorical data becomes numerical
- how train/test splits work
- how Logistic Regression works mathematically
- how binary classification works
- how models are evaluated using accuracy and confusion matrices

---

# 🧠 What Is Machine Learning?

Machine Learning is the process of teaching a computer to find patterns in data.

Instead of manually programming rules like:

```python
if passenger_is_female:
    survive = True
```

we allow the model to learn patterns automatically from historical examples.

The model studies:
- passenger age
- ticket class
- fare price
- gender
- family size
- and more...

Then it learns relationships between those features and survival outcomes.

---

# 🚢 About the Titanic Dataset

The Titanic dataset contains information about passengers aboard the Titanic.

The model tries to answer:

> "Based on passenger information, did this person survive?"

This is called a:

# Binary Classification Problem

Because there are only two possible outputs:

| Output | Meaning |
|---|---|
| 0 | Did Not Survive |
| 1 | Survived |

---

# 📁 Project Structure

```bash
titanic-survival-predictor/
│
├── data/
│   └── titanic.csv
│
├── outputs/
│   ├── model/
│   └── plots/
│
├── src/
│   ├── download_data.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 02-classical-machine-learning/project_1_titanic_survival_predictor
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

# 🔄 Full Machine Learning Pipeline

This project follows the standard ML workflow:

```text
Load Data
   ↓
Clean Data
   ↓
Encode Features
   ↓
Split Dataset
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Make Predictions
```

---

# 📂 Step 1 — Loading Data

File:
```bash
src/data_loader.py
```

The dataset is loaded using Pandas.

```python
df = pd.read_csv(path)
```

This creates a DataFrame:

| Passenger | Age | Sex | Fare | Survived |
|---|---|---|---|---|
| 1 | 22 | male | 7.25 | 0 |

Think of a DataFrame as a spreadsheet inside Python.

---

# 🧹 Step 2 — Data Preprocessing

File:
```bash
src/preprocessing.py
```

This is one of the MOST IMPORTANT steps in machine learning.

Real-world data is messy.

Models cannot learn effectively from:
- missing values
- text categories
- inconsistent formatting

So we preprocess the data first.

---

# 📌 Handling Missing Values

Some passengers are missing ages.

Machine learning models cannot train with missing values.

So we replace missing ages with the median age:

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

Why median instead of average?

Because median is less affected by outliers.

---

# 📌 Encoding Categorical Variables

Machine learning models work with numbers, not text.

Example:

| Sex |
|---|
| male |
| female |

Becomes:

| Sex |
|---|
| 0 |
| 1 |

This process is called:

# Feature Encoding

We use:

```python
LabelEncoder()
```

to convert categories into numerical values.

---

# 📌 Feature Selection

The model only trains on useful columns:

```python
[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]
```

These are called:

# Features

Features are the input variables used by the model to make predictions.

---

# 📌 Target Variable

The value we want to predict is:

```python
Survived
```

This is called the:

# Target Variable

---

# 📌 Train/Test Split

The dataset is split into:

| Dataset | Purpose |
|---|---|
| Training Set | Teach the model |
| Test Set | Evaluate the model |

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.2
)
```

80% is used for training.

20% is held back for testing.

This is critical because:

> We must evaluate the model on data it has NEVER seen before.

Otherwise the model may simply memorize.

---

# 🤖 Step 3 — Logistic Regression

File:
```bash
src/train.py
```

This project uses:

# Logistic Regression

Despite the name, Logistic Regression is actually a:

# Classification Algorithm

It predicts probabilities between 0 and 1.

---
# 🧠 How Logistic Regression Actually Works

Logistic Regression is one of the simplest and most important machine learning algorithms.

Even though modern AI uses deep neural networks, many of the same core ideas begin here.

At its core, Logistic Regression tries to answer:

> "How likely is this example to belong to a certain class?"

In this project:

```text
Class 0 → Did Not Survive
Class 1 → Survived
```

The model outputs a probability between 0 and 1.

---

# 📌 Step 1 — The Model Looks At Features

Each passenger has numerical features:

| Feature | Example |
|---|---|
| Age | 22 |
| Fare | 7.25 |
| Sex | 0 |
| Pclass | 3 |

The model combines all these values into a single score.

---

# 📌 Step 2 — Weighted Sum

The model computes:

```math
z = w_1x_1 + w_2x_2 + w_3x_3 + \dots + w_nx_n + b
```

Where:

| Symbol | Meaning |
|---|---|
| \(x\) | feature value |
| \(w\) | learned weight |
| \(b\) | bias term |
| \(z\) | raw prediction score |

---

# 📌 What Are Weights?

Weights represent:

> how important each feature is.

Example:

| Feature | Weight |
|---|---|
| Sex | +3.2 |
| Age | -0.04 |
| Fare | +0.01 |

Interpretation:

- a positive weight increases survival probability
- a negative weight decreases survival probability

---

# 📌 Intuition Behind Weights

Suppose the model learns:

```text
female passengers survived more often
```

Then the weight for:

```text
Sex = female
```

becomes strongly positive.

If the model learns:

```text
older passengers survived less often
```

then the age weight may become negative.

The model automatically discovers these statistical relationships from data.

---

# 📌 Example Weighted Sum Calculation

Suppose a passenger has:

| Feature | Value |
|---|---|
| Age | 22 |
| Fare | 50 |
| Sex | 1 |

And the model learned:

| Weight | Value |
|---|---|
| Age weight | -0.03 |
| Fare weight | +0.02 |
| Sex weight | +2.5 |
| Bias | -1.2 |

The model computes:

```math
z = (-0.03)(22) + (0.02)(50) + (2.5)(1) - 1.2
```

Step-by-step:

```text
= -0.66 + 1.0 + 2.5 - 1.2
= 1.64
```

This value:

```text
z = 1.64
```

is called the:

# Logit (Raw Score)

But this is NOT yet a probability.

It could be:
- negative
- very large
- any real number

We must convert it into a probability.

---

# 📌 Step 3 — The Sigmoid Function

The raw score is passed through the sigmoid function:

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

Where:

-  \( sigma(z) \) = the output probability (final prediction between 0 and 1)  
- \( z \) = the raw model score (also called the **logit**)  
- \( e \) = Euler’s number (~2.718), a mathematical constant  

And the value of \( z \) is computed as:

```math
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
```

Where:

- \( x_1, x_2, ... , x_n \) = input features (Age, Fare, Sex, etc.)  
- \( w_1, w_2, ... , w_n \) = learned weights (importance of each feature)  
- \( b \) = bias term (adjustment constant)  


The sigmoid function transforms ANY number into a value between:

```text
0 and 1
```

which makes it perfect for probabilities.

---

# 📌 Why Sigmoid Is Important

Without sigmoid:

```text
Prediction = 14.7
Prediction = -8.2
```

These are not valid probabilities.

After sigmoid:

| Raw Score \(z\) | Probability |
|---|---|
| -10 | 0.00004 |
| -2 | 0.12 |
| 0 | 0.50 |
| 2 | 0.88 |
| 10 | 0.99995 |

The sigmoid "squashes" values into probability space.

---

# 📌 Example Sigmoid Calculation

Earlier we computed:

```text
z = 1.64
```

Now apply sigmoid:

```math
\sigma(1.64)=\frac{1}{1+e^{-1.64}}
```

Result:

```text
≈ 0.84
```

Meaning:

```text
84% probability of survival
```

---

# 📌 Turning Probability Into A Class

The model finally applies a threshold.

Usually:

```text
Probability > 0.5 → Survived
Probability < 0.5 → Did Not Survive
```

Example:

| Probability | Prediction |
|---|---|
| 0.84 | Survived |
| 0.17 | Did Not Survive |

---

# 📌 How The Model Learns The Weights

Initially, weights are random.

Example:

```text
Age weight = 0.002
Fare weight = -0.1
```

The model makes predictions.

Then it compares predictions against actual answers.

Example:

| Actual | Predicted |
|---|---|
| 1 | 0.23 |
| 0 | 0.89 |

Clearly these are bad predictions.

So the model adjusts weights slightly.

---

# 📌 Optimization

The model repeatedly:

```text
1. predict
2. measure error
3. adjust weights
4. improve
```

This process is called:

# Optimization

The algorithm used is usually:

# Gradient Descent

Gradient Descent tries to reduce prediction error step-by-step.

---

# 📌 The Loss Function

The model needs a way to measure:

> "How wrong am I?"

This is called the:

# Loss Function

For Logistic Regression, we often use:

# Log Loss / Cross-Entropy Loss

The loss becomes:
- small when predictions are correct
- large when predictions are wrong

The optimizer tries to minimize this loss.

---

# 📌 Decision Boundary

Eventually the model learns a decision boundary.

Example:

```text
If probability > 0.5
    classify as survived
else
    classify as died
```

The model learns where this boundary should exist in feature space.

---

# 📌 Why Logistic Regression Is Powerful

Even though it is mathematically simple, Logistic Regression teaches:

- probabilities
- optimization
- feature weighting
- classification
- decision boundaries
- gradient descent
- loss minimization

These are foundational ideas used throughout modern AI.

Even deep neural networks still rely on:
- weighted sums
- activation functions
- optimization
- loss minimization

---

# 📌 Important Intuition

Logistic Regression is NOT memorizing passengers.

It is learning statistical patterns like:

```text
"Passengers with these characteristics
were more likely to survive."
```

Machine learning is fundamentally:
- pattern recognition
- statistical optimization
- probability estimation

---

# 🧠 Final Mental Model

You can think of Logistic Regression as:

```text
features
   ↓
weighted scoring system
   ↓
sigmoid probability conversion
   ↓
classification decision
```

Or more visually:

```text
Passenger Features
        ↓
Weighted Sum
        ↓
Sigmoid Function
        ↓
Survival Probability
        ↓
Final Prediction
```

That basic structure is one of the foundations of machine learning.

---

# 📌 Training The Model

Training means:

> adjusting the model weights to reduce prediction error.

The model repeatedly:
1. makes predictions
2. compares predictions to real answers
3. adjusts weights
4. improves over time

This is optimization.

---

# 📌 Why Logistic Regression Is Great For Beginners

It teaches:
- classification
- probabilities
- feature importance
- optimization
- decision boundaries

while remaining mathematically understandable.

---

# 📊 Step 4 — Model Evaluation

File:
```bash
src/evaluate.py
```

After training, we evaluate performance.

---

# 📌 Accuracy

Accuracy measures:

```text
correct predictions / total predictions
```

Example:

| Actual | Predicted |
|---|---|
| Survived | Survived |
| Died | Died |

If the model predicts correctly 82 out of 100 times:

```text
Accuracy = 82%
```

---

# 📌 Confusion Matrix

A confusion matrix shows:
- correct predictions
- incorrect predictions

Example:

| | Predicted Survived | Predicted Died |
|---|---|---|
| Actually Survived | TP | FN |
| Actually Died | FP | TN |

This helps diagnose model weaknesses.

The project automatically saves a confusion matrix plot.

---

# 📊 Outputs Generated

After running the project:

```bash
outputs/
│
├── model/
│   └── titanic_model.pkl
│
└── plots/
    └── confusion_matrix.png
```

---

# 💾 Saved Model

The trained model is saved using:

```python
joblib.dump()
```

This allows you to:
- reload the model later
- deploy it
- use it for predictions

without retraining.

---

# 🧠 Key Machine Learning Concepts Learned

| Concept | Learned |
|---|---|
| Binary Classification | ✅ |
| Logistic Regression | ✅ |
| Data Cleaning | ✅ |
| Feature Engineering | ✅ |
| Train/Test Split | ✅ |
| Missing Value Handling | ✅ |
| Categorical Encoding | ✅ |
| Model Evaluation | ✅ |
| Accuracy Metrics | ✅ |
| Confusion Matrix | ✅ |

---

# 🚀 Beginner Improvements To Try

Once the basic project works, try adding:

## Easy
- probability prediction
- ROC curve
- feature importance visualization

## Intermediate
- Random Forest classifier
- XGBoost
- cross-validation

## Advanced
- FastAPI deployment
- Docker support
- web dashboard

---

# 🧠 Most Important Lesson

Machine learning is NOT magic.

A model only learns from:
- patterns in data
- feature relationships
- statistical optimization

The most important skill in ML is often:

> understanding and preparing data correctly.

---

# 📚 Recommended Next Projects

After this project:

1. House Price Predictor (Regression)
2. Spam Email Classifier (Text ML)
3. Customer Churn Predictor
4. Stock Movement Baseline Predictor
5. Neural Networks with PyTorch

---

# 🎯 Final Takeaway

This project teaches the foundation of nearly all machine learning systems:

```text
data
   ↓
preprocessing
   ↓
training
   ↓
evaluation
   ↓
prediction
```

Once you understand this workflow, you can begin building much more advanced AI systems.
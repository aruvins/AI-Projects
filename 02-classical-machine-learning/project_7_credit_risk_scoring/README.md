# 🟡 Credit Risk Scoring Model

A beginner-friendly machine learning project that predicts whether a borrower is likely to default on a loan.

This project is designed to teach the foundations of **classical machine learning**, especially how AI is used in **financial risk modeling**.

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 02-classical-machine-learning/project_7_credit_risk_scoring

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

# 🧠 Introduction

Credit risk scoring is one of the most common real-world applications of machine learning.

Banks and lenders must answer a critical question:

> “Will this person repay their loan?”

Machine learning models learn patterns from historical borrower data and estimate the probability that someone will default.

Instead of manually reviewing every applicant, models can automatically analyze:

* income
* debt
* credit history
* payment behavior
* loan size
* financial ratios

and estimate financial risk.

---

# 💳 What Is Credit Risk Scoring?

Credit scoring is the process of assigning a **risk level** to a borrower.

Example:

| Borrower | Predicted Risk        |
| -------- | --------------------- |
| Person A | 3% chance of default  |
| Person B | 78% chance of default |

The lender can then:

* approve safe loans
* reject risky loans
* adjust interest rates
* reduce financial losses

This is a **binary classification problem**:

| Label | Meaning    |
| ----- | ---------- |
| 0     | No Default |
| 1     | Default    |

The model predicts one of these two outcomes.

---

# 🤖 Machine Learning Concepts

This project introduces several core AI and machine learning concepts.

---

## 1. Supervised Learning

This project uses **supervised learning**.

That means:

* we already know the correct answers
* the dataset contains labels

Example:

| Income | Debt Ratio | Defaulted? |
| ------ | ---------- | ---------- |
| 80k    | 0.2        | No         |
| 25k    | 0.9        | Yes        |

The model learns relationships between:

* inputs (features)
* outputs (labels)

---

## 2. Classification

This is specifically a **classification** task.

Classification means:

* predicting categories
* not continuous numbers

Examples:

* spam vs not spam
* fraud vs legitimate
* default vs no default

---

## 3. Features

Features are the input variables used by the model.

Example financial features:

* annual income
* age
* loan amount
* credit score
* missed payments
* debt ratio

The model tries to discover patterns connecting these features to loan default risk.

---

# 📊 Dataset Overview

The project uses a synthetic dataset generated with Scikit-learn.

Example features:

| Feature         | Description             |
| --------------- | ----------------------- |
| income          | Borrower income         |
| loan_amount     | Requested loan size     |
| debt_ratio      | Debt relative to income |
| credit_score    | Creditworthiness        |
| missed_payments | Payment history         |
| savings         | Amount saved            |

Target column:

| Column  | Meaning       |
| ------- | ------------- |
| default | 1 = defaulted |

---

# ⚙️ Project Pipeline

The machine learning workflow follows these steps:

```text id="fj90wh"
Data Collection
      ↓
Data Preprocessing
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Probability Prediction
      ↓
Evaluation Metrics
      ↓
Threshold Tuning
```

This is the standard workflow used in many real-world ML systems.

---

# 📈 Logistic Regression Explained

Logistic Regression is one of the most important classical machine learning algorithms.

Despite its name:

* it is used for classification
* not regression

Its goal is to estimate probabilities.

---

## Linear Combination

The model first computes a weighted sum:

z=w_1x_1+w_2x_2+\cdots+b

Where:

* (x) = features
* (w) = learned weights
* (b) = bias/intercept

Each feature contributes differently to the prediction.

Example:

* missed payments may increase risk heavily
* higher income may reduce risk

---

## Sigmoid Function

The output is transformed into a probability using the sigmoid function:

P(y=1)=\frac{1}{1+e^{-z}}

This converts any value into a probability between:

* 0 and 1

Example:

* 0.02 → low risk
* 0.95 → high risk

---

## Why Logistic Regression Matters

Logistic Regression is widely used because it is:

* simple
* interpretable
* fast
* stable
* easy to explain to regulators

Banks often prefer interpretable models over black-box models.

---

# 🌲 Gradient Boosting Explained

Gradient Boosting is a more advanced machine learning technique.

It builds many small decision trees sequentially.

Each new tree tries to fix the mistakes of previous trees.

---

## Key Idea

Instead of building one large model:

```text id="0guh3t"
Weak Model 1
     ↓ improves errors
Weak Model 2
     ↓ improves errors
Weak Model 3
     ↓ improves errors
Final Strong Model
```

The model gradually improves prediction quality.

---

## Why It Works Well

Gradient Boosting can:

* capture nonlinear relationships
* model complex interactions
* detect subtle patterns

Example:

* high debt may only be dangerous when income is low

Linear models struggle with these interactions.

---

# 🎯 Probability Estimation

Unlike basic classifiers, these models estimate probabilities.

Example:

| Borrower | Default Probability |
| -------- | ------------------- |
| A        | 0.08                |
| B        | 0.71                |

This is extremely important in finance because decisions depend on risk levels.

---

# 📉 ROC-AUC Explained

ROC-AUC is one of the most important classification metrics.

It measures:

> how well the model separates positive and negative classes.

---

## ROC Curve

The ROC curve plots:

| Axis   | Meaning             |
| ------ | ------------------- |
| X-axis | False Positive Rate |
| Y-axis | True Positive Rate  |

---

## AUC Score

AUC stands for:

> Area Under the Curve

Interpretation:

| AUC   | Meaning         |
| ----- | --------------- |
| 0.50  | Random guessing |
| 0.70  | Decent          |
| 0.80  | Strong          |
| 0.90+ | Excellent       |

A higher AUC means the model ranks risky borrowers better.

---

# 🎚 Threshold Tuning

Most classifiers use a default threshold of:

```python id="hm9s1m"
0.5
```

Meaning:

* probability ≥ 0.5 → predict default

But businesses often customize this threshold.

---

## Example

### Conservative lender

Threshold:

```python id="vjlwm4"
0.30
```

This catches more risky borrowers.

---

## Aggressive lender

Threshold:

```python id="6j78w8"
0.70
```

This approves more loans but increases risk.

---

# 📦 Confusion Matrix

A confusion matrix shows prediction outcomes.

|                  | Predicted Safe | Predicted Default |
| ---------------- | -------------- | ----------------- |
| Actually Safe    | True Negative  | False Positive    |
| Actually Default | False Negative | True Positive     |

---

## Why False Negatives Matter

In credit risk:

* False Negative =
  predicting someone is safe when they default

This can cause large financial losses.

Therefore lenders often prioritize:

* recall
* risk detection

over raw accuracy.

---

# 📏 Feature Scaling

Some ML models perform better when features are normalized.

This project uses:

```python id="mylsnx"
StandardScaler()
```

Standardization transforms data so features have:

* mean ≈ 0
* standard deviation ≈ 1

---

## Why Scaling Helps

Without scaling:

| Feature    | Range      |
| ---------- | ---------- |
| income     | 0 → 100000 |
| debt_ratio | 0 → 1      |

Large features dominate smaller ones.

Scaling prevents this imbalance.

---

# ⚠️ Imbalanced Data

In real financial datasets:

* most people repay loans
* only a small percentage default

Example:

| Class      | Percentage |
| ---------- | ---------- |
| No Default | 90%        |
| Default    | 10%        |

This creates an imbalanced dataset.

---

## Why This Is Dangerous

A model could predict:

> “Nobody defaults”

and still achieve:

* 90% accuracy

but be useless.

That is why metrics like:

* ROC-AUC
* precision
* recall
* F1-score

matter more than accuracy alone.

---

# 🏋️ Training Workflow

The training process works like this:

---

## Step 1 — Split Data

```python id="jlwm9m"
train_test_split()
```

The model trains on:

* training data

and evaluates on:

* unseen test data

This measures generalization.

---

## Step 2 — Learn Patterns

The model adjusts internal parameters to reduce prediction errors.

This process is called:

> optimization

---

## Step 3 — Predict Probabilities

After training:

* the model predicts default probabilities

---

## Step 4 — Evaluate Performance

Metrics determine how well the model performs.

---

# 🏦 Real-World Applications

Credit risk models are used by:

* banks
* mortgage lenders
* fintech startups
* insurance companies
* credit card providers

---

## Related AI Systems

Very similar techniques are used for:

* fraud detection
* insurance risk
* spam filtering
* medical diagnosis
* customer churn prediction

---

# 🚀 Future Improvements

Possible upgrades for this project:

---

## 1. Real Financial Datasets

Use:

* LendingClub
* German Credit Dataset
* Kaggle financial datasets

---

## 2. Better Feature Engineering

Create:

* debt-to-income ratio
* credit utilization
* loan-to-income ratio

Feature engineering is often more important than model selection.

---

## 3. More Advanced Models

Try:

* Random Forest
* XGBoost
* LightGBM
* CatBoost

---

## 4. Hyperparameter Tuning

Optimize:

* learning rate
* tree depth
* regularization

using:

* GridSearchCV
* RandomizedSearchCV

---

## 5. Model Explainability

Use:

* SHAP values
* feature importance
* partial dependence plots

to understand model behavior.

---


# 📚 Key Takeaways

By completing this project, you will learn:

* supervised learning
* binary classification
* probability estimation
* logistic regression
* gradient boosting
* ROC-AUC
* threshold tuning
* feature scaling
* imbalanced classification
* model evaluation

These concepts form the foundation of many real-world classical AI systems.

---

# 🧠 Final Thoughts

This project is an excellent introduction to applied machine learning because it combines:

* mathematics
* statistics
* business decisions
* AI modeling
* real-world risk analysis

Even today, many production financial systems still rely heavily on classical machine learning models like:

* Logistic Regression
* Gradient Boosting

because they are:

* reliable
* interpretable
* efficient
* highly effective.

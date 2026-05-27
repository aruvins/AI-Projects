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

# ## 1. Supervised Learning

This project uses **supervised learning**, which is one of the most common types of machine learning.

The word **“supervised”** means:

> the model learns using examples that already contain the correct answers.

Think of it like learning with an answer key.

---

# 🧠 The Core Idea

We give the model:

1. **Inputs (Features)**
   Information about a borrower.

2. **Correct Outputs (Labels)**
   Whether the borrower eventually defaulted.

The model studies many examples and tries to discover patterns connecting:

* borrower characteristics
* repayment outcomes

---

# 📊 Example Dataset

| Income | Debt Ratio | Credit Score | Missed Payments | Defaulted? |
| ------ | ---------- | ------------ | --------------- | ---------- |
| 80k    | 0.2        | 760          | 0               | No         |
| 25k    | 0.9        | 520          | 5               | Yes        |
| 120k   | 0.1        | 810          | 0               | No         |
| 40k    | 0.7        | 590          | 3               | Yes        |

---

# 🔍 What the Model Notices

After seeing thousands of examples, the model may learn patterns like:

### Lower risk patterns

* higher income
* lower debt ratio
* higher credit score
* fewer missed payments

### Higher risk patterns

* low income
* high debt
* poor credit history
* many missed payments

The model does **not** understand finance like a human.

Instead, it learns:

> statistical relationships between numbers and outcomes.

---

# 📦 Features vs Labels

In supervised learning, data is split into:

| Type     | Meaning                    |
| -------- | -------------------------- |
| Features | Inputs used for prediction |
| Labels   | Correct answers            |

---

## Features (Inputs)

These are the variables the model uses to make predictions.

Example features:

* income
* age
* debt ratio
* credit score

Mathematically, features are often written as:

X=[x_1,x_2,x_3,...,x_n]

Where:

* (x_1) might be income
* (x_2) might be debt ratio
* etc.

---

## Labels (Outputs)

Labels are the correct answers the model tries to predict.

For this project:

y\in{0,1}

Where:

* `0` = no default
* `1` = default

---

# 🏋️ How Training Works

During training:

1. The model looks at borrower data
2. Makes a prediction
3. Compares prediction to the real answer
4. Adjusts itself to reduce future mistakes

This process repeats thousands of times.

Over time, the model improves.

---

# 🔄 Simple Example Walkthrough

Imagine the model sees this borrower:

| Feature         | Value |
| --------------- | ----- |
| Income          | 30k   |
| Debt Ratio      | 0.85  |
| Credit Score    | 540   |
| Missed Payments | 4     |

The model predicts:

```text id="18g02q"
Default Probability = 0.82
```

Meaning:

> an 82% chance the borrower will default.

If the borrower actually defaulted:

* the prediction was good

If not:

* the model adjusts its internal parameters

---

# 🧠 What the Model Is Actually Learning

The model learns:

* which features matter most
* how strongly they affect risk
* combinations of risky behaviors

For example:

* high debt alone may not be dangerous
* but high debt + low income might be

This is the essence of machine learning:

> discovering patterns automatically from data.

---

# 📈 Why Supervised Learning Is Powerful

Instead of manually writing rules like:

```text id="rkh32w"
IF debt_ratio > 0.8 THEN risky
```

the model learns rules automatically from historical examples.

This allows ML systems to:

* adapt to new data
* capture complex patterns
* improve with more training data

---

# 🌎 Real-World Examples of Supervised Learning

Supervised learning powers many everyday AI systems:

| Application        | Prediction          |
| ------------------ | ------------------- |
| Email spam filters | Spam or not spam    |
| Fraud detection    | Fraud or legitimate |
| Medical diagnosis  | Disease or healthy  |
| Credit scoring     | Default or safe     |
| Image recognition  | Cat or dog          |
| Customer churn     | Leave or stay       |

---

# ⚠️ Important Limitation

Supervised learning is only as good as the data it learns from.

Bad or biased data can produce:

* inaccurate predictions
* unfair decisions
* biased models

This is a major topic in real-world AI systems, especially in finance and healthcare.

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

```math
z=w_1x_1+w_2x_2+\cdots+b
```

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

```math
P(y=1)=\frac{1}{1+e^{-z}}
```

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

# 🌲 Gradient Boosting Explained

Gradient Boosting is one of the most powerful classical machine learning techniques.

It works by combining:

* many small weak models
* into one strong predictive model

Most commonly, those weak models are:

> small decision trees

---

# 🧠 The Big Idea

Instead of training:

* one giant complex model

Gradient Boosting trains:

* many tiny models
* one after another

Each new model focuses on:

> fixing the mistakes made by earlier models.

---

# 📦 Step-by-Step Intuition

Imagine you're trying to predict loan defaults.

---

## Step 1 — First Tree Makes Predictions

The first decision tree is very simple.

Maybe it learns:

```text id="c3g3p5"
If credit_score < 600:
    predict high risk
Else:
    predict low risk
```

This helps somewhat, but it makes many mistakes.

Example:

| Borrower | Actual  | Predicted |
| -------- | ------- | --------- |
| A        | Default | Default ✅ |
| B        | Safe    | Safe ✅    |
| C        | Default | Safe ❌    |
| D        | Safe    | Default ❌ |

The errors are important.

Gradient Boosting now asks:

> “How can we correct these mistakes?”

---

# 🔄 Step 2 — Train Another Tree on the Errors

The second tree does NOT start from scratch.

Instead:

* it focuses heavily on the incorrect predictions.

Maybe it notices:

```text id="8x0yx7"
People with:
- high debt ratio
- AND low income
were being missed
```

So Tree #2 specializes in fixing that pattern.

---

# 🔄 Step 3 — Add Corrections

The predictions from Tree #2 are added to Tree #1.

Now the overall model becomes better.

---

# 🔄 Step 4 — Repeat Many Times

This process repeats:

```text id="ulaxmd"
Tree 1 → basic patterns
Tree 2 → fixes some errors
Tree 3 → fixes remaining errors
Tree 4 → improves harder cases
...
```

Eventually:

* hundreds of tiny trees
* work together

to create a highly accurate model.

---

# 🧠 Why Tiny Trees Work

Each individual tree is usually:

* shallow
* weak
* inaccurate alone

But together:

> they become extremely powerful.

This idea is called:

> ensemble learning

Meaning:

* combining many weak learners
* into one strong learner

---

# 🌳 What Is a Decision Tree?

A decision tree repeatedly splits data using questions.

Example:

```text id="6x9vzl"
Is credit_score < 650?
    ├── YES
    │     Is debt_ratio > 0.7?
    │          ├── YES → High Risk
    │          └── NO  → Medium Risk
    │
    └── NO
          → Low Risk
```

Trees are easy to understand because they mimic human decision-making.

---

# ⚡ Why It’s Called “Boosting”

The term:

> boosting

means:

> turning weak learners into a strong learner.

Each tree “boosts” the performance of the previous ensemble.

---

# 📉 What Does “Gradient” Mean?

This is the more mathematical part.

Gradient Boosting uses:

> gradient descent optimization

to minimize prediction errors.

---

## Error Function

The model has a loss function:

L(y,\hat{y})

Where:

* (y) = actual value
* (\hat{y}) = predicted value

The goal is:

> minimize the loss.

---

# 🧠 Gradient = Direction of Improvement

The gradient tells the model:

> “Which direction reduces error fastest?”

This is similar to:

* walking downhill on a mountain
* always stepping downward

Gradient Boosting uses gradients to determine:

* how to improve predictions
* what mistakes to focus on next

---

# 📦 Simplified Mathematical Flow

The model prediction evolves like this:

Initial prediction:

```math
F_0(x)
```

After first correction:

```math
F_1(x)=F_0(x)+h_1(x)
```

After second correction:

```math
F_2(x)=F_1(x)+h_2(x)
```

Eventually:

```math
F_M(x)=\sum_{m=1}^{M} h_m(x)
```
Where:

* each $(h_m(x))$ is a small corrective tree

---

# 🎯 Why Gradient Boosting Is So Powerful

Gradient Boosting can learn:

* nonlinear patterns
* interactions between variables
* complex decision boundaries

---

# 📌 Example of Nonlinear Behavior

Suppose risk behaves like this:

| Income | Debt Ratio | Risk      |
| ------ | ---------- | --------- |
| High   | High       | Medium    |
| Low    | High       | VERY High |

The effect of debt depends on income.

This interaction is difficult for linear models.

Gradient Boosting handles this naturally.

---

# 📈 Gradient Boosting vs Logistic Regression

| Logistic Regression    | Gradient Boosting       |
| ---------------------- | ----------------------- |
| Linear relationships   | Nonlinear relationships |
| Simpler                | More powerful           |
| Very interpretable     | Harder to interpret     |
| Faster                 | Slower                  |
| Lower overfitting risk | Higher overfitting risk |
| Great baseline         | Often higher accuracy   |

---

# ⚠️ Overfitting

Gradient Boosting is powerful enough to:

> memorize training data

if not controlled properly.

This is called:

> overfitting

The model performs:

* extremely well on training data
* poorly on new data

---

# 🛡 How We Prevent Overfitting

Important parameters help regularize the model.

---

## 1. Learning Rate

Controls how much each tree contributes.

Small learning rates:

* learn slowly
* generalize better

Example:

```python id="vyljlwm"
GradientBoostingClassifier(
    learning_rate=0.05
)
```

---

## 2. Number of Trees

More trees:

* more learning capacity
* more overfitting risk

Example:

```python id="o3u3tz"
n_estimators=100
```

---

## 3. Tree Depth

Controls tree complexity.

Shallow trees:

* generalize better
* reduce memorization

Example:

```python id="1q7d4p"
max_depth=3
```

---

# 🚀 Why Gradient Boosting Dominated Classical ML

Before deep learning became popular, Gradient Boosting models dominated:

* Kaggle competitions
* tabular datasets
* finance
* insurance
* recommendation systems

Even today:

* XGBoost
* LightGBM
* CatBoost

are among the best algorithms for structured/tabular data.

---

# 🧠 Intuition Summary

Gradient Boosting works like this:

```text id="3a1foz"
1. Make predictions
2. Find mistakes
3. Train small tree to fix mistakes
4. Add corrections
5. Repeat many times
```

Each tree becomes:

* a specialist at fixing remaining errors

And together:

* the ensemble becomes highly accurate.

---

# 🌎 Real-World Uses

Gradient Boosting is used for:

* credit risk scoring
* fraud detection
* insurance pricing
* ad click prediction
* recommendation systems
* customer churn prediction
* search ranking

It remains one of the most important classical machine learning techniques ever created.

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

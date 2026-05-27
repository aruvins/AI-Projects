# 🔵 Model Comparison & Benchmarking System

A beginner-friendly machine learning project that compares multiple classical ML models using:

* cross-validation
* pipelines
* hyperparameter tuning
* visualization
* benchmarking metrics

This project is designed to teach one of the most important real-world ML skills:

> how to evaluate machine learning models scientifically.

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 02-classical-machine-learning/project_8_model_comparison

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

In machine learning, training a model is only part of the process.

A much bigger challenge is:

> determining which model actually performs best.

Different ML algorithms behave differently depending on:

* dataset structure
* feature relationships
* noise
* scale
* complexity

This project creates a reusable benchmarking framework that:

* trains multiple models
* evaluates them fairly
* visualizes results
* compares performance scientifically

---

# ❓ Why Model Comparison Matters

There is no universally best machine learning algorithm.

Example:

* Logistic Regression works well on linear problems
* Random Forest handles nonlinear relationships
* SVMs work well in high-dimensional spaces
* KNN relies heavily on local structure

The only reliable way to choose a model is:

> empirical evaluation.

That means:

* testing models
* measuring results
* comparing metrics

This process is called:

> benchmarking.

---

# 📊 What Is Benchmarking?

Benchmarking means:

> systematically comparing models using the same dataset and evaluation process.

A proper benchmark ensures:

* fair comparison
* reproducibility
* reliable conclusions

Without benchmarking:

* results can be misleading
* models may appear better than they actually are

---

# 🧠 Supervised Learning

This project uses:

> supervised learning

Meaning:

* the dataset already contains correct answers (labels)

Example:

| Features                    | Label     |
| --------------------------- | --------- |
| tumor size, radius, texture | malignant |
| tumor size, smoothness      | benign    |

The model learns:

* relationships between inputs
* and outputs

---

# 🎯 Classification Problems

This project solves a:

> classification problem

Classification means:

* predicting categories

Example:

| Problem           | Classes            |
| ----------------- | ------------------ |
| Spam Detection    | spam / not spam    |
| Fraud Detection   | fraud / legitimate |
| Medical Diagnosis | benign / malignant |

In this project:

* models classify tumors as:

  * malignant
  * benign

---

# ⚙️ Machine Learning Pipelines

One of the most important engineering concepts in ML is the:

> pipeline

A pipeline automates:

* preprocessing
* transformations
* model training

Example:

```python id="w4mjlwm"
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

---

# 🧠 Why Pipelines Matter

Without pipelines:

* preprocessing mistakes are common
* scaling may accidentally leak information
* workflows become messy

Pipelines guarantee:

* consistent preprocessing
* reproducible workflows
* safer experimentation

---

# ⚠️ Data Leakage Explained

Data leakage is one of the most dangerous ML mistakes.

It happens when:

> information from the test set accidentally influences training.

This creates:

* unrealistically high performance
* misleading evaluation

---

## Example Leakage Problem

Incorrect workflow:

```text id="0k0onm"
1. Scale ALL data
2. Split train/test
```

The scaler already saw the test data.

This leaks information.

---

## Correct Workflow

```text id="r8h0ot"
1. Split train/test
2. Fit scaler ONLY on training data
3. Transform test data
```

Pipelines automate this correctly.

---

# 🔄 Cross-Validation Explained 

Cross-validation is one of the most important concepts in machine learning evaluation.

Its purpose is to answer this question:

> “How well will this model perform on unseen data?”

A single train/test split often gives unreliable answers because performance can vary depending on:

* randomness
* dataset composition
* outliers
* class distribution

Cross-validation reduces this problem by evaluating the model multiple times on different subsets of the data.

---

# 🧠 The Core Problem With One Train/Test Split

Suppose you split data like this:

```text id="f90cxq"
80% → Training
20% → Testing
```

You train the model once and get:

```text id="jjlwm4"
Accuracy = 96%
```

Looks great.

But what if:

* the test set happened to be unusually easy?
* or unusually difficult?

Then your estimate may be misleading.

---

# 📉 Example of Split Variability

Imagine your dataset has:

* 1000 samples
* some difficult edge cases
* some noisy data

If your test split accidentally contains:

* mostly easy examples

your model may appear much better than it truly is.

Another random split might produce:

| Split   | Accuracy |
| ------- | -------- |
| Split A | 96%      |
| Split B | 89%      |
| Split C | 92%      |

Which one is correct?

This uncertainty is exactly why cross-validation exists.

---

# 📦 The Big Idea of Cross-Validation

Instead of testing:

* once

Cross-validation tests:

* many times

using different train/validation splits.

This produces:

* a distribution of scores
* a more reliable performance estimate

---

# 🔄 K-Fold Cross Validation

The most common version is:

> K-Fold Cross Validation

The dataset is divided into:

* (K) equal sections called folds.

Example:

```python id="jlwm7m"
cv=5
```

means:

* 5 folds

---

# 📊 Visualizing 5-Fold Cross Validation

Suppose we have this dataset:

```text id="jlwm3k"
[1][2][3][4][5]
```

Each block is a fold.

---

# Iteration 1

```text id="jlwm5u"
VALIDATE → [1]

TRAIN → [2][3][4][5]
```

---

# Iteration 2

```text id="jlwm2r"
VALIDATE → [2]

TRAIN → [1][3][4][5]
```

---

# Iteration 3

```text id="jlwm6j"
VALIDATE → [3]

TRAIN → [1][2][4][5]
```

---

# Iteration 4

```text id="jlwm1z"
VALIDATE → [4]

TRAIN → [1][2][3][5]
```

---

# Iteration 5

```text id="jlwm0t"
VALIDATE → [5]

TRAIN → [1][2][3][4]
```

---

# 🧠 Important Observation

Every sample gets used:

* for training
* AND for validation

This is powerful because:

* no data is wasted
* evaluation becomes more robust

---

# 📈 Final Cross-Validation Score

Suppose the fold accuracies are:

| Fold | Accuracy |
| ---- | -------- |
| 1    | 0.94     |
| 2    | 0.96     |
| 3    | 0.91     |
| 4    | 0.95     |
| 5    | 0.93     |

Final CV score:

```math
CV\ Score=\frac{0.94+0.96+0.91+0.95+0.93}{5}
```

Result:

```text id="jlwm8v"
Mean CV Accuracy = 0.938
```

This estimate is much more reliable than:

* one random split

---

# 📉 Variance Matters Too

Cross-validation also reveals:

> stability

Suppose:

| Fold Scores |
| ----------- |
| 0.99        |
| 0.98        |
| 0.45        |
| 0.97        |
| 0.99        |

Average:

* looks decent

BUT:

* performance is inconsistent

This suggests:

* instability
* overfitting
* data sensitivity

---

# 📏 Mean vs Standard Deviation

Cross-validation often reports:

| Metric             | Meaning             |
| ------------------ | ------------------- |
| Mean               | Average performance |
| Standard deviation | Stability           |

Example:

```text id="jlwm9n"
Accuracy = 0.94 ± 0.02
```

Meaning:

* average = 94%
* variation is small
* model is stable

---

# 🧠 Why Cross-Validation Is Better

Cross-validation provides:

---

## 1. Better Generalization Estimates

It better approximates:

> real-world unseen performance.

---

## 2. Reduced Randomness

Performance depends less on:

* lucky splits
* unlucky splits

---

## 3. Better Model Comparison

Suppose:

| Model               | Single Split Accuracy |
| ------------------- | --------------------- |
| Logistic Regression | 94%                   |
| Random Forest       | 95%                   |

Looks close.

But cross-validation may reveal:

| Model               | Mean CV Accuracy |
| ------------------- | ---------------- |
| Logistic Regression | 94% ± 1%         |
| Random Forest       | 95% ± 8%         |

The Random Forest is much less stable.

---

# ⚠️ Important: Validation ≠ Test Set

A common beginner mistake:

```text id="jlwm4x"
Train → Validation → Test
```

Cross-validation happens ONLY inside:

* training/validation

The final test set should remain untouched until the very end.

---

# 🧪 Proper Workflow

Correct process:

```text id="jlwm2c"
1. Split train/test
2. Perform cross-validation on training set
3. Choose best model
4. Evaluate ONCE on test set
```

This prevents:

* overfitting to the test set

---

# ⚡ Computational Cost

Cross-validation is more expensive.

Example:

| Method       | Number of Trainings |
| ------------ | ------------------- |
| Single Split | 1                   |
| 5-Fold CV    | 5                   |
| 10-Fold CV   | 10                  |

If hyperparameter tuning is also used:

```text id="jlwm7f"
100 parameter combinations
×
5 folds
=
500 trainings
```

This becomes computationally heavy.

---

# 🔍 Stratified Cross Validation

In classification problems, we often use:

```python id="jlwm3a"
StratifiedKFold
```

This preserves:

* class distribution in each fold

Example:

| Dataset | 90% Negative / 10% Positive |

Each fold keeps:

* roughly the same ratio

This is critical for:

* imbalanced datasets

---

# 📦 Leave-One-Out Cross Validation (LOOCV)

Extreme version:

```text id="jlwm5m"
Train on ALL samples except 1
Validate on that 1 sample
Repeat for every sample
```

If dataset size = 1000:

* 1000 trainings

Very accurate:

* but extremely slow

---

# 🧠 Bias vs Variance Tradeoff

Choosing (K) affects evaluation behavior.

---

## Smaller K (e.g. 3)

* faster
* higher bias
* less stable

---

## Larger K (e.g. 10)

* slower
* lower bias
* better estimates

Common choices:

* 5-fold
* 10-fold

---

# 🚨 Common Beginner Mistakes

---

## 1. Data Leakage

Scaling BEFORE cross-validation:

❌ Wrong:

```python id="jlwm1e"
scaler.fit_transform(X)
cross_val_score(...)
```

The scaler saw all data.

---

## Correct:

Use pipelines:

```python id="jlwm0s"
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

---

# 2. Using Test Data Repeatedly

If you keep checking test performance:

* you indirectly optimize for the test set

This contaminates evaluation.

---

# 3. Ignoring Variance

Two models with similar averages may have:

* very different stability

Always consider:

* mean
* standard deviation

---

# 🌎 Real-World Importance

Cross-validation is used heavily in:

* Kaggle competitions
* academic ML research
* production ML systems
* medical AI
* financial modeling

Without proper validation:

* models can fail catastrophically in production.

---

# 🧠 Intuition Summary

Cross-validation works like this:

```text id="jlwm8p"
1. Split data into folds
2. Train on most folds
3. Validate on remaining fold
4. Repeat many times
5. Average the results
```

Instead of asking:

> “Did the model perform well once?”

Cross-validation asks:

> “Does the model perform well consistently across many different subsets of the data?”

---

# 🏆 Model Benchmarking

This project benchmarks several ML algorithms:

| Model               | Type              |
| ------------------- | ----------------- |
| Logistic Regression | Linear            |
| SVM                 | Margin-based      |
| Random Forest       | Bagging ensemble  |
| Gradient Boosting   | Boosting ensemble |
| KNN                 | Distance-based    |

Each model has:

* different strengths
* different weaknesses

---

# 📈 Logistic Regression

Logistic Regression predicts probabilities using:

```math
P(y=1)=\frac{1}{1+e^{-z}}
```

Where:

* (z) is a weighted combination of features

It works well for:

* linear relationships
* interpretable models
* fast training

---

# 🌲 Random Forest

Random Forest builds:

* many decision trees

Each tree:

* trains on random subsets
* makes predictions independently

Final prediction:

> majority vote across trees

This reduces:

* overfitting
* variance

---

# 🌲 Gradient Boosting

Gradient Boosting builds trees sequentially.

Each new tree:

> corrects mistakes made by previous trees.

This creates:

* highly accurate ensembles
* strong nonlinear modeling capability

---

# 📏 K-Nearest Neighbors (KNN)

KNN classifies based on nearby examples.

Example:

* if most nearby points are malignant
* classify as malignant

KNN relies on:

> distance in feature space.

---

# 📐 Support Vector Machines (SVM)

SVMs try to find:

> the optimal separating boundary between classes.

The goal:

* maximize separation margin

This often generalizes very well.

---

# 🎛 Hyperparameter Tuning

Machine learning models contain:

> hyperparameters

These are settings chosen BEFORE training.

Example:

```python id="ngmjlwm"
RandomForestClassifier(
    n_estimators=100,
    max_depth=5
)
```

---

# 🧠 Hyperparameters vs Parameters

| Type            | Learned? |
| --------------- | -------- |
| Parameters      | Yes      |
| Hyperparameters | No       |

Example:

* tree weights → learned
* max_depth → manually chosen

---

# 🔍 Grid Search Explained

Grid Search automatically tests:

* many hyperparameter combinations

Example:

```text id="jlwm9y"
max_depth:
    3
    5
    10

n_estimators:
    50
    100
```

The system evaluates:

> every combination.

Then selects:

* the best-performing configuration

---

# 📊 Evaluation Metrics

Accuracy alone is often misleading.

This project evaluates multiple metrics.

---

# ✅ Accuracy

Measures overall correctness:

```math
Accuracy=\frac{Correct\ Predictions}{Total\ Predictions}
```

---

# 🎯 Precision

Precision measures:

> How many predicted positives were actually correct?

High precision means:

* fewer false positives

---

# 🔎 Recall

Recall measures:

> How many actual positives were detected?

High recall means:

* fewer missed positives

---

# ⚖️ F1 Score

F1 balances:

* precision
* recall

```math
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
```

Useful for:

* imbalanced datasets
* medical problems
* fraud detection

---

# 📉 ROC Curves

ROC curves evaluate:

* classification quality across thresholds

The curve plots:

| Axis   | Meaning             |
| ------ | ------------------- |
| X-axis | False Positive Rate |
| Y-axis | True Positive Rate  |

---

# 📈 ROC-AUC

AUC =

> Area Under the Curve

Interpretation:

| AUC  | Quality         |
| ---- | --------------- |
| 0.50 | Random guessing |
| 0.80 | Strong          |
| 0.95 | Excellent       |

Higher AUC means:

* better class separation

---

# 📦 Confusion Matrix

Confusion matrices show prediction breakdowns.

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | True Positive      | False Negative     |
| Actual Negative | False Positive     | True Negative      |

This helps diagnose:

* model weaknesses
* error types

---

# 📏 Feature Scaling

Some algorithms rely heavily on feature magnitudes.

Example:

* KNN
* SVM
* Logistic Regression

This project uses:

```python id="z8djlwm"
StandardScaler()
```

Standardization transforms features so they have:

* mean ≈ 0
* standard deviation ≈ 1

---

# ⚠️ Overfitting vs Underfitting

---

## Underfitting

The model is:

* too simple
* unable to capture patterns

Results:

* poor training performance
* poor test performance

---

## Overfitting

The model memorizes training data.

Results:

* excellent training performance
* poor generalization

Cross-validation helps detect this.

---

# 📊 Visualization System

The project automatically generates:

* ROC curves
* confusion matrices
* model comparison charts
* CSV benchmark reports

This simulates real-world ML experimentation pipelines.

---

# 📁 Outputs Folder

Generated outputs:

```text id="0ujm6w"
outputs/
├── benchmark_results.csv
├── model_comparison.png
├── roc_curves.png
├── confusion_matrix_*.png
```

---

# 🌎 Real-World Applications

Benchmarking systems are used in:

* ML research
* Kaggle competitions
* production AI systems
* automated ML pipelines
* model monitoring frameworks

Large companies benchmark hundreds of models before deployment.

---

# 🚀 Future Improvements

Possible upgrades:

---

## Add More Models

Try:

* XGBoost
* LightGBM
* CatBoost

---

## Experiment Tracking

Add:

* MLflow
* Weights & Biases

---

## Automatic Reports

Generate:

* HTML dashboards
* PDF summaries

---

## Feature Importance

Visualize:

* which features influence predictions most

---

## Regression Benchmarking

Support:

* regression tasks
* forecasting problems

---

# 🧠 Final Thoughts

This project teaches one of the most important practical ML skills:

> evaluating models correctly.

Many beginners focus only on:

* training models

But real-world ML engineering depends heavily on:

* benchmarking
* validation
* reproducibility
* scientific comparison

These skills are foundational for:

* machine learning engineering
* data science
* AI research
* production ML systems.

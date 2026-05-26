# 🌸 Iris Flower Classifier

A beginner-friendly machine learning project that classifies iris flower species using multiple ML models.

This project focuses on understanding:

* how machine learning models work
* how to train and evaluate models
* how different algorithms make decisions
* how to visualize datasets and predictions

---

## 📁 Project Structure

```
project_project_4_iris_flower_classifier/
│
├── output/
│   ├── decision_tree_confusion_matrix.png
│   ├── decision_tree_dataset_visualization.png
│   ├── knn_confusion_matrix.png
│   ├── knn_dataset_visualization.png
│   ├── svm_confusion_matrix.png
│   ├── svm_dataset_visualization.png
│   ├── model_accuracy_comparison.png
│   └── results.txt
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 02-classical-machine-learning/project_4_iris_flower_classifier
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

# 📚 What This Project Teaches

## Core Machine Learning Concepts

### 1. Train/Test Split

Machine learning models need:

* training data → to learn patterns
* testing data → to evaluate performance

We split the dataset into:

```text
80% Training Data
20% Testing Data
```

This helps measure how well the model performs on unseen data.

---

### 2. Feature Scaling

Machine learning models often perform better when all features are on a similar numerical scale.

For example:

| Feature | Example Values |
|---|---|
| Petal Length | 1.0 - 6.9 |
| Petal Width | 0.1 - 2.5 |

Without scaling, features with larger values can dominate distance calculations.

We use:

```python
StandardScaler()
```

to standardize each feature using the formula:

```math
z = \frac{x - \mu}{\sigma}
```

Where:

* (x) = original value
* (mu) = mean of the feature
* (sigma) = standard deviation of the feature

After scaling:
- the mean becomes approximately 0
- the standard deviation becomes 1

This ensures every feature contributes more equally during training.

Scaling is especially important for:
- KNN → uses distance calculations
- SVM → builds optimal decision boundaries based on feature space distances

Without scaling, models may become biased toward features with larger numeric ranges.

---

### 3. Classification

The goal is to predict which flower species an iris belongs to.

Possible classes:

* setosa
* versicolor
* virginica

This is called:

> Multi-class classification

---

# 🌺 The Iris Dataset

The Iris dataset is one of the most famous beginner ML datasets.

It contains flower measurements for 3 iris species.

---

## Features

Each flower has 4 measurements:

| Feature      | Description          |
| ------------ | -------------------- |
| Sepal Length | Length of outer leaf |
| Sepal Width  | Width of outer leaf  |
| Petal Length | Length of petal      |
| Petal Width  | Width of petal       |

---

## Classes

| Species    |
| ---------- |
| Setosa     |
| Versicolor |
| Virginica  |

---

# 🤖 Models Used

---

# 1. K-Nearest Neighbors (KNN)

## Idea

KNN predicts by looking at nearby data points.

Think of it like:

> “Which flowers are closest to this new flower?”

The model checks the nearest neighbors and chooses the most common class.

---

## Example

If the 3 nearest flowers are:

```text
Setosa
Setosa
Versicolor
```

The prediction becomes:

```text
Setosa
```

because it appears most often.

---

## How It Works

### Step 1

Store all training data.

### Step 2

When a new flower appears:

* calculate distance to all known flowers

### Step 3

Find the nearest K neighbors.

### Step 4

Use majority vote.

---

## Visualization

KNN creates smooth regions around groups of similar flowers.

Nearby points usually share the same prediction.

---

## Pros

✅ Easy to understand
✅ Works well on small datasets
✅ No training phase needed

---

## Cons

❌ Slower on large datasets
❌ Sensitive to scaling
❌ Can struggle with noisy data

---

# 2. Decision Tree

## Idea

A Decision Tree makes predictions using rules.

It repeatedly asks questions like:

```text
Is petal length < 2.5?
```

Each answer moves down the tree until a prediction is reached.

---

## Example

```text
Petal Length < 2.5?
    YES → Setosa
    NO →
        Petal Width < 1.8?
            YES → Versicolor
            NO → Virginica
```

---

## How It Works

The model:

1. Finds the best feature to split data
2. Creates branches
3. Repeats recursively

The goal is to separate classes as clearly as possible.

---

## Visualization

Decision Trees create rectangular decision regions.

The boundaries are sharp and rule-based.

---

## Pros

✅ Very interpretable
✅ Easy to visualize
✅ Handles non-linear patterns

---

## Cons

❌ Can overfit
❌ Sensitive to small data changes
❌ Deep trees become unstable

---

# 3. Support Vector Machine (SVM)

## Idea

SVM tries to find the best boundary between classes.

It searches for a line (or hyperplane) that maximizes separation.

---

## Example

Imagine separating two flower species with the widest possible gap:

```text
Class A  | margin |  boundary  | margin |  Class B
```

SVM chooses the boundary with the largest margin.

---

## How It Works

### Step 1

Plot all data points.

### Step 2

Find the optimal separating boundary.

### Step 3

Maximize the distance between classes.

Only the closest points matter most.

These are called:

> Support Vectors

---

## Visualization

SVM creates clean separation boundaries.

It often performs extremely well on structured datasets.

---

## Pros

✅ High accuracy
✅ Powerful on small/medium datasets
✅ Handles complex boundaries well

---

## Cons

❌ Harder to understand mathematically
❌ Slower on large datasets
❌ Sensitive to parameter tuning

---

# 📊 Evaluation Metrics

---

# Accuracy

Measures:

> How many predictions were correct?

Formula:

```math
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
```

Example:

```text
27 correct out of 30
Accuracy = 90%
```

---

# Precision

Measures:

> When the model predicts a class, how often is it correct?

---

# Recall

Measures:

> How many actual examples did the model successfully find?

---

# F1 Score

Balances:

* precision
* recall

Useful when classes are imbalanced.

---

# 📈 Visualizations

The project generates several visual outputs.

---

## 1. Confusion Matrix

Shows:

* correct predictions
* incorrect predictions

Example:

| Actual    | Predicted    |
| --------- | ------------ |
| Setosa    | Setosa ✅     |
| Virginica | Versicolor ❌ |

---

## 2. Decision Boundary Visualization

The dataset is reduced to 2D using:

```python
PCA (Principal Component Analysis)
```

The visualization shows:

* flower data points
* model prediction regions
* class boundaries

This helps understand:

* how each model separates classes
* how predictions are made

---

# 📁 Output Folder

Generated images are saved in:

```text
output/
```

Example files:

```text
output/
├── decision_tree_confusion_matrix.png
├── decision_tree_dataset_visualization.png
│
├── knn_confusion_matrix.png
├── knn_dataset_visualization.png
│
├── svm_confusion_matrix.png
├── svm_dataset_visualization.png
│
├── model_accuracy_comparison.png
│
└── results.txt
```

# 🧠 Example Output

```text
========== KNN ==========
Accuracy: 1.0000

========== Decision Tree ==========
Accuracy: 0.9667

========== SVM ==========
Accuracy: 1.0000
```

---

# 🚀 Possible Improvements

You can expand this project with:

* hyperparameter tuning
* cross validation
* GridSearchCV
* additional classifiers
* real-world datasets
* interactive visualizations
* model persistence with pickle/joblib

---

# 🎯 Final Goal

Understand how machine learning classifiers:

* learn patterns
* separate classes
* make predictions
* evaluate performance

while building clean ML project structure.

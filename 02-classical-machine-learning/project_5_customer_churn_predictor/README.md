# 📊 Customer Churn Predictor

A beginner-friendly machine learning project that predicts whether a customer will leave a telecommunications company (churn) using a Random Forest model.

This project focuses on understanding:

* how machine learning models work on real-world business data
* how to handle imbalanced datasets
* how tabular categorical variables are processed
* how to extract and visualize feature importance

---

## 📁 Project Structure

```
project_5_customer_churn_predictor/
│
├── output/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── results.txt
│
├── main.py
├── telco-customer-churn.csv
├── requirements.txt
└── README.md

```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 02-classical-machine-learning/project_5_customer_churn_predictor
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

### 1. Train/Test Split with Stratification

Machine learning models need separate data chunks to learn patterns and evaluate performance. We split the dataset into:

```text
80% Training Data
20% Testing Data

```

Because churn datasets are typically highly imbalanced (far more customers stay than leave), we use **stratification** (`stratify=y`). This ensures that both the training set and testing set contain the exact same percentage of churned vs. retained customers, preventing skewed evaluation scores.

---

### 2. Categorical Encoding (One-Hot Encoding)

Machine learning algorithms can only process numbers, but real business data contains text columns like `Contract` ("Month-to-month", "One year") or `InternetService` ("DSL", "Fiber optic").

We use:

```python
pd.get_dummies(X, drop_first=True)

```

This converts text variables into numeric column flags (0 or 1). Dropping the first category prevents multi-collinearity (redundant data) while keeping the dataset clean for the model.

---

### 3. Feature Scaling

While tree-based models like Random Forest do not strictly require scaling, we apply a `StandardScaler` to normalize features with vastly different ranges (such as `Tenure` in months vs. `TotalCharges` in thousands of dollars).

We standardize each feature using the formula:

$$z = \frac{x - \mu}{\sigma}$$

Where:

* $x$ = original value
* $\mu$ = mean of the feature
* $\sigma$ = standard deviation of the feature

This ensures data consistency across standard machine learning workflows.

---

### 4. Binary Classification & Imbalance Handling

The goal is to predict one of two states for a customer:

* Churn (1)
* No Churn (0)

This is a **Binary Classification** task. To stop the model from simply guessing "No Churn" every time to achieve a fake high accuracy, we activate:

```python
class_weight="balanced"

```

This forces the algorithm to penalize mistakes on the smaller "Churn" group heavily, making it much more sensitive to finding high-risk accounts.

---

# 🏢 The Telco Customer Churn Dataset

This dataset contains behavior info for thousands of simulated telecom users.

---

## Core Features Used

| Feature Component | Examples |
| --- | --- |
| **Demographics** | Gender, Senior Citizen status, Partner, Dependents |
| **Account Info** | Tenure (months), Contract type, Paperless billing, Payment method |
| **Services Signed Up** | Phone, Multiple lines, Fiber optic/DSL, Online security, Tech support |
| **Financials** | Monthly charges, Total charges |

---

# 🤖 The Model: Random Forest

## Idea

A Random Forest Classifier predicts behavior by combining the outputs of multiple decision makers. Think of it like a panel of diverse experts voting on a customer's risk profile.

---

## How It Works

Instead of building one massive, strict decision rulebook, a Random Forest builds hundreds of individual **Decision Trees** (this project uses `n_estimators=200`).

### Step 1: Subsampling

Each individual tree is trained on a slightly different random slice of the customer data.

### Step 2: Feature Splitting

At every branch split, the tree can only select from a randomized subset of features (e.g., assessing `Contract` type or `MonthlyCharges`).

### Step 3: Majority Voting

When predicting if a new customer will leave, all 200 trees cast their votes. The final probability and classification are based on the collective consensus.

---

## Pros

✅ Excellent at handling highly non-linear business patterns
✅ Built-in protection against overfitting (via random ensembles)
✅ Uncovers which variables matter most to your business

## Cons

❌ Harder to instantly visualize compared to a single decision tree
❌ Predictions can be slightly slower when scaled up to millions of rows

---

# 📊 Evaluation Metrics

---

# Classification Report

Because accuracy is misleading on imbalanced datasets, we monitor three primary metrics:

### Precision

> When the model flags a customer as a churn risk, how often is it right?

### Recall (Sensitivity)

> Out of all the customers who actually left, what percentage did the model successfully catch?

### F1-Score

> The harmonic balance of Precision and Recall.

---

# ROC-AUC Score

The **Receiver Operating Characteristic - Area Under Curve** measures how well the model separates the true churners from the safe customers across all possible probability thresholds. A score of `0.5` is as bad as a random coin flip, while `1.0` is a flawless predictor.

---

# 📈 Visualizations

---

## 1. Confusion Matrix (`output/confusion_matrix.png`)

This grid provides a breakdown of your model's exact accuracy classifications:

| Row (Actual) | Column (Predicted: No Churn) | Column (Predicted: Churn) |
| --- | --- | --- |
| **Actual No Churn** | **True Negatives (TN):** Correctly predicted to stay | **False Positives (FP):** Safe users flagged as risks |
| **Actual Churn** | **False Negatives (FN):** Missed retention targets | **True Positives (TP):** Correctly caught churn risks |

---

## 2. Top Feature Importance (`output/feature_importance.png`)

This horizontal bar chart ranks which specific traits (e.g., total charges, short-term contracts, absence of tech support) carried the most weight across all 200 internal decision trees.

---

# 🧠 Example Output

```text
📊 Customer Churn Prediction Results

              precision    recall  f1-score   support

          No       0.84      0.89      0.87      1035
         Yes       0.64      0.53      0.58       374

    accuracy                           0.80      1409
   macro avg       0.74      0.71      0.73      1409
weighted avg       0.79      0.80      0.79      1409

ROC-AUC Score: 0.8322
Saved confusion matrix to: output/confusion_matrix.png
Saved feature importance to: output/feature_importance.png

🔮 Sample Churn Probabilities:
Customer 1: 0.50% chance of churn
Customer 2: 63.00% chance of churn
Customer 3: 6.00% chance of churn
Customer 4: 37.50% chance of churn
Customer 5: 1.00% chance of churn
Customer 6: 59.00% chance of churn
Customer 7: 47.50% chance of churn
Customer 8: 3.00% chance of churn
Customer 9: 0.00% chance of churn
Customer 10: 50.00% chance of churn
```

---

# 🚀 Possible Improvements

You can expand this project by applying:

* **Hyperparameter Tuning:** Use `GridSearchCV` to optimize `max_depth` and `min_samples_split`.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Generate synthetic churn profiles instead of relying solely on class weight balancing.
* **Alternative Classifiers:** Compare the performance against gradient-boosted architectures like XGBoost or LightGBM.
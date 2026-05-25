# 🏠 Project 2 — House Price Predictor

## 🎯 Purpose

This project introduces **regression modeling**, one of the most important concepts in machine learning.

Unlike classification (yes/no outcomes), regression predicts **continuous values**, such as:

- house prices
- temperature
- stock prices
- demand forecasting

Here, we predict:

> 💰 The price of a house based on its features

---

# 🧠 What You Will Learn

By completing this project, you will understand:

- what regression is and how it differs from classification
- how models learn relationships between variables
- how to preprocess real-world data
- how to evaluate predictions using error metrics
- how to visualize model performance

---

# 📊 Dataset Overview

We use a housing dataset containing features such as:

| Feature | Meaning |
|---|---|
| median_income | average income in the area |
| housing_median_age | age of houses |
| total_rooms | number of rooms |
| total_bedrooms | number of bedrooms |
| population | number of people |
| households | number of households |
| ocean_proximity | location category |
| median_house_value | 🧠 target (what we predict) |

---

# 🧩 Problem Type

This is a:

```text
Regression Problem
```

Because the output is a continuous number:

```text
$452,000
$123,500
$980,000
```

---

# ⚙️ Project Pipeline

The machine learning workflow looks like this:

```text
Data Loading
   ↓
Data Cleaning
   ↓
Feature Encoding
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Visualization
```

---

# 📁 Project Structure

```bash
house-price-predictor/
│
├── data/
│   └── housing.csv
│
├── outputs/
│   ├── model/
│   └── plots/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│
├── download_data.py
├── main.py
└── requirements.txt
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 02-classical-machine-learning/project_2_house_price_predictor
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

# 🤖 Models Used

This section explains **how these models actually work internally**, not just what they do.

---

# 1️⃣ Linear Regression (The “Best Fit Line” Model)

## 🧠 Core Idea

Linear Regression tries to answer:

> “What is the best equation that relates inputs to output?”

It assumes the output (house price) can be written as a **weighted sum of features**.

---

## 📌 The Model Equation

```math
y = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
```

Where:
- \(x\) = input features (income, rooms, location, etc.)
- \(w\) = weights (importance of each feature)
- \(b\) = bias (base price adjustment)
- \(y\) = predicted house price

---

## 🧠 Intuition

Think of it like a pricing formula:

```text
House Price =
    (income × importance)
  + (number of rooms × importance)
  + (location factor × importance)
  + base price
```

The model is learning the **best possible weights** for each factor.

---

## 📉 How It Learns

The model starts with random weights, then repeatedly:

1. Makes predictions
2. Compares predictions to real prices
3. Measures error
4. Adjusts weights slightly

This process is called:

> 🔁 Gradient Descent

---

## 📌 What “Best Fit” Means

The model tries to minimize error:

```text
error = actual price - predicted price
```

It adjusts weights until total error is as small as possible.

---

## 📊 Why It’s Called “Linear”

Because it assumes:

> The relationship between inputs and output is a straight-line combination

That means:
- doubling a feature roughly doubles its effect
- no complex interactions between features

---

## ✔ Strengths

- Very fast to train
- Easy to interpret
- Works well for simple relationships

---

## ❌ Weaknesses

- Cannot model complex patterns
- Struggles with:
  - non-linear relationships
  - feature interactions
  - real-world complexity

---

# 2️⃣ Random Forest Regressor (The “Committee of Decision Trees”)

## 🧠 Core Idea

Instead of one formula, Random Forest uses:

> Many decision trees voting together

Each tree makes its own prediction, and the final answer is the **average**.

---

## 🌳 What is a Decision Tree?

A decision tree is like a flowchart:

```text
Is income > 5?
   ├── Yes → Is house size > 2000?
   │          ├── Yes → High price
   │          └── No → Medium price
   └── No → Low price
```

It splits data step-by-step based on conditions.

---

## 🌲 What is a Random Forest?

A Random Forest is:

> Hundreds of decision trees working together

Each tree:
- sees a random sample of data
- sees a random subset of features
- learns slightly different rules

---

## 🧠 Why randomness helps

If one tree makes a mistake, others correct it.

This reduces:
- overfitting
- noise sensitivity
- instability

---

## 📊 Final Prediction

For random forest:

```text
Final prediction =
average of all tree predictions
```

Example:

| Tree | Prediction |
|---|---|
| Tree 1 | 420,000 |
| Tree 2 | 450,000 |
| Tree 3 | 430,000 |

Final result:

```text
433,333
```

---

## 🔍 Why It Works So Well

Random Forest is powerful because it:
- learns non-linear patterns
- captures feature interactions
- is robust to noise

Example:

It can learn rules like:

```text
IF income is high AND location is coastal
THEN price is very high
```

Linear regression cannot easily learn this.

---

## ✔ Strengths

- Very accurate
- Handles complex relationships
- Works well without much tuning

---

## ❌ Weaknesses

- Hard to interpret
- Slower than linear regression
- Larger models

---

# ⚖️ Linear Regression vs Random Forest

| Feature | Linear Regression | Random Forest |
|---|---|---|
| Speed | ⚡ Very fast | Slower |
| Accuracy | Medium | High |
| Interpretability | Easy | Hard |
| Complexity handling | Low | High |
| Overfitting risk | Low | Medium (controlled) |

---

# 🧠 Big Picture Understanding

Both models try to learn:

```text
input features → house price
```

But they do it differently:

### Linear Regression:
> “Let me draw the best equation line”

### Random Forest:
> “Let me ask many decision trees and average their opinions”

---

# 📉 How Evaluation Works

We measure how far predictions are from actual values.

---

## 📌 MAE (Mean Absolute Error)

```text
average absolute difference between predicted and actual values
```

👉 Easy to interpret  
👉 “On average, we are off by $X”

---

## 📌 MSE (Mean Squared Error)

Punishes large mistakes more heavily.

---

## 📌 RMSE (Root Mean Squared Error)

Same as MSE, but in original units (dollars).

---

# 📊 Visualizations (IMPORTANT)

After running the project, plots are saved in:

```bash
outputs/plots/
```

---

## 📌 1. Actual vs Predicted

Shows how close predictions are to real values.

- perfect model → diagonal line
- spread → error

---

## 📌 2. Residual Distribution

Residual =

```text
error = actual - predicted
```

A good model:
- centered around 0
- symmetric distribution

---

## 📌 3. Residual vs Predictions

Helps detect:
- bias
- patterns in errors
- missing features

---

# 🧠 Key Machine Learning Concepts

✔ Regression modeling  
✔ Feature engineering  
✔ Handling missing data  
✔ One-hot encoding  
✔ Train/test splitting  
✔ Model comparison  
✔ Error metrics  
✔ Data visualization  

---

# 🔥 Why This Project Matters

This project teaches a core idea in machine learning:

```text
Data → Model → Prediction → Error → Improvement
```

This loop is the foundation of ALL ML systems.

---

# 🚀 Next Steps (Upgrades)

Once you understand this project, try:

## Beginner+
- add feature scaling (StandardScaler)
- log-transform house prices
- visualize feature importance

## Intermediate
- XGBoost / LightGBM
- cross-validation
- hyperparameter tuning

## Advanced
- deploy model with FastAPI
- build web UI (Streamlit)
- Dockerize the project

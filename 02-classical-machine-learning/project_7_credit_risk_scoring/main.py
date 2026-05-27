import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. Generate Synthetic Credit Data
# =========================================================

X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.85, 0.15],   # 15% defaults
    random_state=42
)

feature_names = [
    "income",
    "age",
    "loan_amount",
    "credit_score",
    "employment_length",
    "debt_ratio",
    "missed_payments",
    "savings",
    "credit_history",
    "open_accounts"
]

df = pd.DataFrame(X, columns=feature_names)
df["default"] = y

print(df.head())

# =========================================================
# 2. Train/Test Split
# =========================================================

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 3. Feature Scaling
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 4. Train Models
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():

    print(f"\n{'='*50}")
    print(name)
    print('='*50)

    # Train
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Probabilities
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

# =========================================================
# 5. Plot ROC Curve
# =========================================================

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()

# =========================================================
# 6. Threshold Tuning
# =========================================================

print("\n" + "="*50)
print("Threshold Tuning Example")
print("="*50)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Custom threshold
threshold = 0.30

custom_predictions = (probabilities >= threshold).astype(int)

print(f"Using threshold: {threshold}")

print(confusion_matrix(y_test, custom_predictions))
print(classification_report(y_test, custom_predictions))
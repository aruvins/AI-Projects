import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# ---------------------------------------------------
# Setup
# ---------------------------------------------------

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
df = pd.read_csv("telco-customer-churn.csv")

# Expected format:
# - features in columns
# - target column named "Churn" (Yes/No or 1/0)

# Convert target if needed
if df["Churn"].dtype == object:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------------------------------------------
# Split features and target
# ---------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Handle categorical variables (simple encoding)
X = pd.get_dummies(X, drop_first=True)

# ---------------------------------------------------
# Train/Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Feature Scaling (optional for trees, but kept for consistency)
# ---------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# Model (handles imbalance well)
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

print("\n📊 Customer Churn Prediction Results\n")

print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_score:.4f}")

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Churn Model")

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"Saved confusion matrix to: {cm_path}")

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------

importances = model.feature_importances_
feature_names = X.columns

top_idx = importances.argsort()[-15:]  # top 15 features

plt.figure(figsize=(8, 6))
plt.barh(feature_names[top_idx], importances[top_idx])
plt.title("Top Feature Importance (Random Forest)")

fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(fi_path)
plt.close()

print(f"Saved feature importance to: {fi_path}")

# ---------------------------------------------------
# Business Insight: Churn Probability Preview
# ---------------------------------------------------

sample_probs = y_prob[:10]

print("\n🔮 Sample Churn Probabilities:")
for i, p in enumerate(sample_probs):
    print(f"Customer {i+1}: {p:.2%} chance of churn")
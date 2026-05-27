import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import (
    SelectKBest,
    f_classif
)

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression

# =========================================================
# 1. Create Outputs Folder
# =========================================================

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. Load Dataset
# =========================================================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)

# =========================================================
# 3. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 4. Define Feature Engineering Experiments
# =========================================================

experiments = {
    "No Scaling": Pipeline([
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "StandardScaler": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "MinMaxScaler": Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Feature Selection + StandardScaler": Pipeline([
        ("feature_selection", SelectKBest(
            score_func=f_classif,
            k=10
        )),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])
}

# =========================================================
# 5. Run Experiments
# =========================================================

results = []

best_model = None
best_f1 = 0

for name, pipeline in experiments.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    # -----------------------------------------------------
    # Cross Validation
    # -----------------------------------------------------

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="f1"
    )

    print(f"Mean CV F1: {cv_scores.mean():.4f}")

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    pipeline.fit(X_train, y_train)

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    y_pred = pipeline.predict(X_test)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    results.append({
        "Experiment": name,
        "CV F1": cv_scores.mean(),
        "Accuracy": accuracy,
        "F1 Score": f1
    })

    # -----------------------------------------------------
    # Save Best Model
    # -----------------------------------------------------

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline

# =========================================================
# 6. Save Best Model
# =========================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "best_model.pkl"
)

joblib.dump(best_model, model_path)

print(f"\nBest model saved to: {model_path}")

# =========================================================
# 7. Create Results DataFrame
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nFinal Results:")
print(results_df)

# =========================================================
# 8. Save Results CSV
# =========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "comparison_results.csv"
)

results_df.to_csv(csv_path, index=False)

# =========================================================
# 9. Model Comparison Plot
# =========================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Experiment",
    y="F1 Score"
)

plt.title("Feature Engineering Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(plot_path)

plt.close()

# =========================================================
# 10. Feature Importance Visualization
# =========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

pipeline.fit(X_train, y_train)

model = pipeline.named_steps["model"]

importance = np.abs(model.coef_[0])

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# ---------------------------------------------------------
# Plot Feature Importance
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

sns.barplot(
    data=feature_importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top Feature Importances")

plt.tight_layout()

importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)

plt.savefig(importance_path)

plt.close()
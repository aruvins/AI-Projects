import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

# =========================================================
# 1. Create Outputs Folder
# =========================================================

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nOutputs will be saved to: {OUTPUT_DIR}/")

# =========================================================
# 2. Load Dataset
# =========================================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("\nDataset Shape:", X.shape)

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
# 4. Define Models
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),

    "SVM": SVC(probability=True),

    "Random Forest": RandomForestClassifier(),

    "Gradient Boosting": GradientBoostingClassifier(),

    "KNN": KNeighborsClassifier()
}

# =========================================================
# 5. Benchmark Models
# =========================================================

results = []

plt.figure(figsize=(10, 7))

for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    # -----------------------------------------------------
    # Pipeline
    # -----------------------------------------------------

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # -----------------------------------------------------
    # Cross Validation
    # -----------------------------------------------------

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    pipeline.fit(X_train, y_train)

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # -----------------------------------------------------
    # Save Results
    # -----------------------------------------------------

    results.append({
        "Model": name,
        "CV Accuracy": cv_scores.mean(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    # =====================================================
    # 6. ROC Curve Visualization
    # =====================================================

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.3f})"
    )

    # =====================================================
    # 7. Confusion Matrix Visualization
    # =====================================================

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(f"Confusion Matrix - {name}")

    filename = f"confusion_matrix_{name.replace(' ', '_')}.png"

    plt.savefig(os.path.join(OUTPUT_DIR, filename))

    plt.close()

# =========================================================
# 8. Save ROC Curves
# =========================================================

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))

plt.close()

# =========================================================
# 9. Create Results DataFrame
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nFinal Results:")
print(results_df)

# =========================================================
# 10. Save CSV Results
# =========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "benchmark_results.csv"
)

results_df.to_csv(csv_path, index=False)

print(f"\nResults saved to: {csv_path}")

# =========================================================
# 11. Model Comparison Visualization
# =========================================================

plt.figure(figsize=(12, 6))

plt.bar(
    results_df["Model"],
    results_df["F1 Score"]
)

plt.ylabel("F1 Score")

plt.title("Model Performance Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(comparison_path)

plt.close()

print(f"Model comparison chart saved to:")
print(comparison_path)

# =========================================================
# 12. Hyperparameter Tuning
# =========================================================

print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

param_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5, 10],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Score:")
print(grid_search.best_score_)

# =========================================================
# 13. Evaluate Best Model
# =========================================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nFinal Tuned Model Metrics:")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

print("\nProject Complete!")
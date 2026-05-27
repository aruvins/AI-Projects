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
    f1_score
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
# 1. Load Dataset
# =========================================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("\nDataset Shape:")
print(X.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# =========================================================
# 2. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 3. Define Models
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),

    "SVM": SVC(probability=True),

    "Random Forest": RandomForestClassifier(),

    "Gradient Boosting": GradientBoostingClassifier(),

    "KNN": KNeighborsClassifier()
}

# =========================================================
# 4. Benchmark Models
# =========================================================

results = []

for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # ---------------------------------------------
    # Cross Validation
    # ---------------------------------------------

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(f"Cross Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

    # ---------------------------------------------
    # Train Model
    # ---------------------------------------------

    pipeline.fit(X_train, y_train)

    # ---------------------------------------------
    # Predictions
    # ---------------------------------------------

    y_pred = pipeline.predict(X_test)

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    results.append({
        "Model": name,
        "CV Accuracy": cv_scores.mean(),
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# =========================================================
# 5. Create Results Table
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

print(results_df.sort_values(by="F1 Score", ascending=False))

# =========================================================
# 6. Hyperparameter Tuning
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

print("\nBest Cross Validation Score:")
print(grid_search.best_score_)

# =========================================================
# 7. Evaluate Best Model
# =========================================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nFinal Tuned Model Metrics:")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

# =========================================================
# 8. Visualization
# =========================================================

plt.figure(figsize=(10, 5))

plt.bar(results_df["Model"], results_df["F1 Score"])

plt.ylabel("F1 Score")
plt.title("Model Comparison")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()
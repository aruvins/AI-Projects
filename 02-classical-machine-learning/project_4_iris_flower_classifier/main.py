from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import os


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

class_names = iris.target_names


# ---------------------------------------------------
# Train/Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# Feature Scaling
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------
# Models
# ---------------------------------------------------

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "SVM": SVC(
        kernel="linear"
    )
}


# ---------------------------------------------------
# Train + Evaluate
# ---------------------------------------------------

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n🌸 Iris Flower Classification Results\n")

pca = PCA(n_components=2)

X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Store accuracies
model_names = []
accuracies = []

for name, model in models.items():
    # Train Model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    model_names.append(name)
    accuracies.append(accuracy)

    print(f"========== {name} ==========")
    print(f"Accuracy: {accuracy:.4f}\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=class_names
        )
    )

    # ---------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------
    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(6, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(ax=ax)

    plt.title(f"{name} Confusion Matrix")

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")

    # ---------------------------------------------------
    # Dataset Visualization + Decision Boundary
    # ---------------------------------------------------
    # Train model on 2D PCA data
    model.fit(X_train_2d, y_train)

    # Create mesh grid
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    # Predict over mesh
    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))

    # Decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)

    # Dataset points
    scatter = plt.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        edgecolors="k"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(f"{name} Decision Boundary on Iris Dataset")

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles=handles,
        labels=class_names.tolist(),
        title="Species"
    )
    visualization_path = os.path.join(
        OUTPUT_DIR,
        f"{name.lower().replace(' ', '_')}_dataset_visualization.png"
    )

    plt.savefig(visualization_path)
    plt.close()

    print(f"Saved dataset visualization to: {visualization_path}\n")


# ---------------------------------------------------
# Accuracy Comparison Chart
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(model_names, accuracies)

plt.ylabel("Accuracy")
plt.ylim(0, 1.05)

plt.title("Model Accuracy Comparison")

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_accuracy_comparison.png"
)

plt.savefig(comparison_path)
plt.close()

print(f"Saved comparison chart to: {comparison_path}")
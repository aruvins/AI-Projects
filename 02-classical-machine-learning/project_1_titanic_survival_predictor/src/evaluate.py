from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt="d")

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("outputs/plots/confusion_matrix.png")

    plt.show()
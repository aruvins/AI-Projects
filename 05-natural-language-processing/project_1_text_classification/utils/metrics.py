from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\nAccuracy")
    print(f"{accuracy:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    return confusion_matrix(
        y_true,
        y_pred
    )
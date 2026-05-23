import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_accuracy(preds, labels):

    """
    Computes classification accuracy.
    """

    preds = torch.argmax(preds, dim=1)

    correct = (preds == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


def top_k_accuracy(preds, labels, k=5):

    """
    Computes Top-K accuracy.
    """

    top_k = torch.topk(
        preds,
        k=k,
        dim=1
    ).indices

    correct = top_k.eq(
        labels.view(-1, 1)
    ).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


def classification_metrics(preds, labels):

    """
    Computes:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    """

    preds = torch.argmax(preds, dim=1)

    preds = preds.cpu().numpy()

    labels = labels.cpu().numpy()

    accuracy = accuracy_score(
        labels,
        preds
    )

    precision = precision_score(
        labels,
        preds,
        average="macro"
    )

    recall = recall_score(
        labels,
        preds,
        average="macro"
    )

    f1 = f1_score(
        labels,
        preds,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def print_metrics(metrics):

    """
    Nicely prints metrics dictionary.
    """

    print("\nEvaluation Metrics")

    print("-" * 30)

    for key, value in metrics.items():

        print(f"{key}: {value:.4f}")
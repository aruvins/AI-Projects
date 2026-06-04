import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch


# =====================================================
# LOSS CURVE
# =====================================================

def plot_loss_curve(
    losses,
    save_path="outputs/training_loss.png"
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        losses,
        marker="o"
    )

    plt.title(
        "Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()


# =====================================================
# ACCURACY CURVE
# =====================================================

def plot_accuracy_curve(
    accuracies,
    save_path="outputs/accuracy_curve.png"
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        accuracies,
        marker="o"
    )

    plt.title(
        "Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()


# =====================================================
# CONFUSION MATRIX
# =====================================================

def plot_confusion_matrix(
    model,
    loader,
    class_names,
    device,
    save_path="outputs/confusion_matrix.png"
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plt.figure(figsize=(10, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

def plot_predictions(model, loader, class_names, device, save_path="outputs/predictions.png"):
    model.eval()
    images, labels = next(
        iter(loader)
    )
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(
            dim=1
        )

    images = images.cpu()
    preds = preds.cpu()

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6)
    )
    axes = axes.flatten()

    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    for i in range(8):

        image = images[i].permute(
            1,
            2,
            0
        ).numpy()

        image = (
            image * std
            + mean
        )

        image = np.clip(
            image,
            0,
            1
        )

        axes[i].imshow(image)

        axes[i].set_title(
            f"P: {class_names[preds[i]]}\n"
            f"T: {class_names[labels[i]]}"
        )

        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
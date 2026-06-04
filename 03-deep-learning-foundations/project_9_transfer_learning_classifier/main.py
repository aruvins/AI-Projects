import os

import torch
import torch.nn as nn
import torch.optim as optim

from models.transfer_model import TransferClassifier

from utils.dataset import get_dataloaders

from utils.trainer import (
    train_epoch,
    evaluate
)

from utils.visualize import (
    plot_loss_curve,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_predictions
)

# =====================================================
# CONFIG
# =====================================================

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_PATH = "outputs/transfer_model.pth"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# OUTPUT DIRECTORY
# =====================================================

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("Transfer Learning Image Classifier")
print(f"Using device: {DEVICE}")
print("=" * 60)

# =====================================================
# DATASET
# =====================================================

print("\nLoading CIFAR-10 Dataset...\n")

train_loader, test_loader, classes = get_dataloaders(
    batch_size=BATCH_SIZE
)

print(f"Number of Classes: {len(classes)}")
print(f"Classes: {classes}")

# =====================================================
# MODEL
# =====================================================

model = TransferClassifier(
    num_classes=len(classes)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

# =====================================================
# LOAD OR TRAIN
# =====================================================

losses = []
accuracies = []

if os.path.exists(MODEL_PATH):

    print("\nLoading saved model...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    print("Model loaded successfully.")

else:

    print("\nNo saved model found.")
    print("Starting training...\n")

    for epoch in range(EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )

        accuracy = evaluate(
            model,
            test_loader,
            DEVICE
        )

        losses.append(train_loss)
        accuracies.append(accuracy)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    print("\nSaving model...")

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(f"Saved -> {MODEL_PATH}")

# =====================================================
# VISUALIZATIONS
# =====================================================

print("\nGenerating Visualizations...")

if len(losses) > 0:

    plot_loss_curve(
        losses,
        save_path="outputs/training_loss.png"
    )

    plot_accuracy_curve(
        accuracies,
        save_path="outputs/accuracy_curve.png"
    )

else:

    print(
        "Skipping loss/accuracy plots "
        "(loaded existing model)."
    )

plot_confusion_matrix(
    model,
    test_loader,
    classes,
    DEVICE,
    save_path="outputs/confusion_matrix.png"
)

plot_predictions(
    model,
    test_loader,
    classes,
    DEVICE,
    save_path="outputs/predictions.png"
)

# =====================================================
# SUMMARY
# =====================================================

print("\nGenerated Files:")

if len(losses) > 0:

    print("  outputs/training_loss.png")
    print("  outputs/accuracy_curve.png")

print("  outputs/confusion_matrix.png")
print("  outputs/predictions.png")
print("  outputs/transfer_model.pth")

print("\nProject Complete.")
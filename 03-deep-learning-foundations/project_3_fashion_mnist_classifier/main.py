import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from utils.model import FashionClassifier


# --------------------------------------------------
# Create output folder
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)
MODEL_PATH = "outputs/fashion_classifier.pth"

# -------------------------------------------------
# Device
# -------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# -------------------------------------------------
# Dataset
# -------------------------------------------------
# Define the transformation to convert images to tensors
transform = transforms.ToTensor()

dataset = datasets.FashionMNIST(
    root = "data",
    train=True,
    download=True,
    transform=transform
)

# 80/20 train-validation split
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = FashionClassifier().to(DEVICE)

# --------------------------------------------------
# Training
# --------------------------------------------------

if os.path.exists(MODEL_PATH):

    print("Existing trained model found!")
    print("Loading saved model...")

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

else:

    print("No trained model found.")
    print("Training new model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    EPOCHS = 10
    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                predictions = outputs.argmax(1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        val_accuracies.append(accuracy)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {accuracy:.4f}"
        )

    # -------------------------------------------------
    # Plot training loss
    # -------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/training_loss.png")
    plt.show()

    # --------------------------------------------------
    # Validation Accuracy Curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("outputs/validation_accuracy.png")
    plt.show()

    # -------------------------------------------------
    # Save the trained model
    # -------------------------------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to", MODEL_PATH)


# --------------------------------------------------
# Test Evaluation
# --------------------------------------------------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)

        preds = outputs.argmax(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

test_accuracy = (
    np.array(all_preds) ==
    np.array(all_labels)
).mean()

print(f"\nTest Accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

plt.savefig("outputs/confusion_matrix.png")
plt.show()

# --------------------------------------------------
# Sample Predictions
# --------------------------------------------------

class_names = [
    "T-Shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

images, labels = next(iter(test_loader))

images = images[:9]
labels = labels[:9]

with torch.no_grad():

    outputs = model(images.to(DEVICE))

predictions = outputs.argmax(1).cpu()

fig, axes = plt.subplots(
    3,
    3,
    figsize=(8, 8)
)

for i, ax in enumerate(axes.flat):

    ax.imshow(
        images[i].squeeze(),
        cmap="gray"
    )

    ax.set_title(
        f"P:{class_names[predictions[i]]}"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig("outputs/sample_predictions.png")
plt.show()

print("\nVisualizations saved to outputs/")
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.model import CNN

# ------------------------------------------------
# Setup
# ------------------------------------------------
os.makedirs("outputs", exist_ok=True)
MODEL_PATH = "outputs/cifar_model.pth"

# ------------------------------------------------
# Device
# ------------------------------------------------
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")

# ------------------------------------------------
# Data
# ------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = dataset.classes

# ------------------------------------------------
# Model
# ------------------------------------------------
model = CNN().to(DEVICE)

# ------------------------------------------------
# Load or Train
# ------------------------------------------------
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

else:
    print("Training model...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10
    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ---------------- Validation ----------------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        val_accuracies.append(acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    # ------------------------------------------------
    # Loss Curve
    # ------------------------------------------------
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("outputs/training_loss.png")
    plt.show()

    # ------------------------------------------------
    # Accuracy Curve
    # ------------------------------------------------
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("outputs/validation_accuracy.png")
    plt.show()

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

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

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

print(f"\nTest Accuracy: {accuracy:.4f}")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot()

plt.savefig("outputs/confusion_matrix.png")
plt.show()

# --------------------------------------------------
# Sample Predictions
# --------------------------------------------------
dataiter = iter(test_loader)
images, labels = next(dataiter)

images_gpu = images.to(DEVICE)
outputs = model(images_gpu)
_, predicted = torch.max(outputs, 1)

# move to CPU for plotting
images = images.cpu().numpy()

fig, axes = plt.subplots(5, 6, figsize=(12, 5))
axes = axes.ravel()

for i in range(30):
    img = np.transpose(images[i] / 2 + 0.5, (1, 2, 0))

    axes[i].imshow(img, interpolation="nearest")
    axes[i].set_title(f"P: {classes[predicted[i]]}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("outputs/predictions.png")
plt.show()
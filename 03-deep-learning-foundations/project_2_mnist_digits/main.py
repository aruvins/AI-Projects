import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.model import DigitClassifier

# -------------------------------------------------
# Create output directory
# -------------------------------------------------

os.makedirs("outputs", exist_ok=True)
MODEL_PATH = "outputs/digit_classifier.pth"

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
# Data Transforms
# -------------------------------------------------
# Define the transformation to convert images to tensors
transform = transforms.ToTensor()

# -------------------------------------------------
# Load Datasets
# -------------------------------------------------
train_dataset = datasets.MNIST(
    root="data", 
    train=True, 
    transform=transform, 
    download=True
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform,
    download=True
)

# -------------------------------------------------
# Data Loaders
# -------------------------------------------------
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# -------------------------------------------------
# Initialize model, loss function and optimizer
# -------------------------------------------------
model = DigitClassifier().to(DEVICE)


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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    EPOCHS = 5
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for images, labels in train_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_epoch_loss:.4f}")

    # -------------------------------------------------
    # Save the trained model
    # -------------------------------------------------
    torch.save(model.state_dict(), "outputs/digit_classifier.pth")
    print("Model saved to outputs/digit_classifier.pth")

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

# -------------------------------------------------
# Evaluate the model on the test set
# -------------------------------------------------
model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# -------------------------------------------------
# Visualize some predictions
# -------------------------------------------------
def visualize_predictions(model, dataloader, device, num_images=5):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Predicted: {predicted[i]}, True: {labels[i]}")
        plt.axis("off")
    plt.savefig("outputs/predictions.png")
    plt.show()

visualize_predictions(model, test_dataloader, DEVICE)
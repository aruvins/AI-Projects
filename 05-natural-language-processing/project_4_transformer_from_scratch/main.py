import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer import TransformerClassifier

from utils.dataset import create_dataloaders

from utils.trainer import (
    train_model,
    evaluate
)

from utils.visualization import (
    plot_attention,
    plot_loss,
    plot_accuracy,
    plot_confusion_matrix
)

# -----------------------------------------
# Setup
# -----------------------------------------

OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "transformer_model.pth"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")

# -----------------------------------------
# Load Data
# -----------------------------------------

print("\nLoading dataset...")

train_loader, test_loader, vocab = (
    create_dataloaders()
)

print(f"Vocabulary Size: {len(vocab):,}")

# -----------------------------------------
# Initialize Model
# -----------------------------------------

model = TransformerClassifier(
    vocab_size=len(vocab),
    embed_size=128,
    num_classes=2
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------------
# Load Existing Model
# -----------------------------------------

if os.path.exists(MODEL_PATH):

    print(
        f"\nFound trained model:"
        f"\n{MODEL_PATH}"
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    print("Model loaded successfully.")

    losses = []
    accuracies = []

else:

    print(
        "\nNo trained model found."
        "\nStarting training..."
    )

    losses, accuracies, attention_weights, cm = (
        train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            epochs=5
        )
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )

# -----------------------------------------
# Evaluation
# -----------------------------------------

print("\nEvaluating model...")

accuracy, cm, attention_weights = evaluate(
    model,
    test_loader,
    device
)

print(
    f"\nFinal Test Accuracy:"
    f" {accuracy:.4f}"
)

# -----------------------------------------
# Visualizations
# -----------------------------------------

print("\nGenerating visualizations...")

if len(losses) > 0:
    plot_loss(losses)

if len(accuracies) > 0:
    plot_accuracy(accuracies)

plot_attention(attention_weights)

plot_confusion_matrix(
    cm,
    ["Negative", "Positive"]
)

print(
    "\nSaved outputs:"
    "\n- attention_map.png"
    "\n- confusion_matrix.png"
)

if len(losses) > 0:
    print("- training_loss.png")

if len(accuracies) > 0:
    print("- validation_accuracy.png")

print("\nDone.")
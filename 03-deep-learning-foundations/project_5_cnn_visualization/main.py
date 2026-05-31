import torch
import torch.nn as nn
import torch.optim as optim
import os

from utils.dataset import get_dataloaders
from utils.model import CNN
from utils.trainer import train_epoch
from utils.visualize import visualize

# Ensure output directory exists
os.makedirs("./outputs", exist_ok=True)

MODEL_PATH = "./outputs/cnn_model.pth"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

train_loader, test_loader = get_dataloaders()

model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -------------------------------
# TRAIN ONLY IF NO MODEL EXISTS
# -------------------------------
if os.path.exists(MODEL_PATH):
    print("✅ Model found. Skipping training...")
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

else:
    print("⚠️ No model found. Training model...")

    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):

        loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )
        print(f"Epoch {epoch + 1}: {loss:.4f}")
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"💾 Model saved to {MODEL_PATH}")


# -------------------------------
# ALWAYS RUN VISUALIZATION
# -------------------------------
visualize()
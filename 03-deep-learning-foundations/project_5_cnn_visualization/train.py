import torch
import torch.nn as nn
import torch.optim as optim
import os

from utils.dataset import get_dataloaders
from utils.model import CNN
from utils.trainer import train_epoch

os.makedirs("./outputs", exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

train_loader, test_loader = get_dataloaders()

model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

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

torch.save(model.state_dict(), "outputs/cnn_model.pth")
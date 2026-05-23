import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.vit_model import ViTClassifier

from utils.dataset import (
    train_loader,
    test_loader
)

# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


model = ViTClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4
)

EPOCHS = 5


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader)

    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()

        loop.set_description(
            f"Epoch [{epoch+1}/{EPOCHS}]"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    print(
        f"Epoch Loss: {running_loss / len(train_loader):.4f}"
    )
from utils.metrics import (
    calculate_accuracy,
    classification_metrics,
    print_metrics
)

metrics = classification_metrics(
    outputs,
    labels
)

print_metrics(metrics)
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for texts, labels in progress_bar:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    return total_loss / len(loader), correct / total

def train_model(model, loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    losses = []
    accuracies = []

    for epoch in range(epochs):
        loss, acc = train_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device
        )

        losses.append(loss)
        accuracies.append(acc)

        print(
            f"Epoch {epoch+1}: "
            f"Loss={loss:.4f} "
            f"Acc={acc:.4f}"
        )

    return losses, accuracies


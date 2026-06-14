import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0

    for inputs, labels in loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    predictions = []
    targets = []

    attention_map = None

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)

            outputs, attention = model(inputs)

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

            if attention_map is None:

                attention_map = (
                    attention[0]
                    .cpu()
                    .numpy()
                )

    acc = accuracy_score(
        targets,
        predictions
    )

    cm = confusion_matrix(
        targets,
        predictions
    )

    return (
        acc,
        cm,
        attention_map
    )

def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    epochs=10
):

    losses = []
    accuracies = []

    attention_map = None
    cm = None

    for epoch in range(epochs):

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        acc, cm, attention_map = evaluate(
            model,
            test_loader,
            device
        )

        losses.append(loss)
        accuracies.append(acc)

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" | Loss: {loss:.4f}"
            f" | Accuracy: {acc:.4f}"
        )

    return (
        losses,
        accuracies,
        attention_map,
        cm
    )
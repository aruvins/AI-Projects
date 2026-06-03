import torch


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    """
    Train for one epoch.

    Returns:
        average loss
    """

    model.train()

    total_loss = 0.0

    for texts, labels in loader:

        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(texts)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(loader)
    )

    return average_loss


def evaluate(
    model,
    loader,
    device
):
    """
    Evaluate classification accuracy.

    Returns:
        accuracy percentage
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for texts, labels in loader:

            texts = texts.to(device)
            labels = labels.to(device)

            outputs, _ = model(texts)

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = (
        100 * correct / total
    )

    return accuracy
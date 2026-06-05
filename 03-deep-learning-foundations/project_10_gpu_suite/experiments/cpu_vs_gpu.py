import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNN
from utils.timer import Timer
from utils.metrics import throughput


def benchmark_device(device, train_loader):
    # Train a simple CNN for one epoch and measure the time taken and throughput
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3,)
    timer = Timer()

    model.train()
    timer.start()
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_samples += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = timer.stop()

    return {
        "device": str(device),
        "epoch_time": elapsed,
        "throughput": throughput(
            total_samples,
            elapsed
        )
    }
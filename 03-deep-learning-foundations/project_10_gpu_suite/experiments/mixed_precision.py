import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNN
from utils.timer import Timer
from utils.metrics import throughput


def benchmark_fp32(train_loader,device,):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    timer = Timer()
    model.train()
    total_samples = 0
    timer.start()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()
        total_samples += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = timer.stop()

    return {
        "mode": "fp32",
        "epoch_time": elapsed,
        "throughput": throughput(
            total_samples,
            elapsed
        )
    }

def benchmark_amp(train_loader, device):
    if device.type != "cuda":
        raise RuntimeError(
            "AMP requires CUDA."
        )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    scaler = torch.amp.GradScaler("cuda")
    timer = Timer()
    model.train()
    total_samples = 0
    timer.start()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_samples += labels.size(0)

    torch.cuda.synchronize()
    elapsed = timer.stop()

    return {
        "mode": "amp",
        "epoch_time": elapsed,
        "throughput": throughput(
            total_samples,
            elapsed
        )
    }
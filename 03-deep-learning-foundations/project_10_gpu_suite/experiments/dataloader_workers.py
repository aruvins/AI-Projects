import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from models.cnn import CNN
from utils.timer import Timer


def benchmark_workers(dataset, worker_counts, batch_size, device):
    results = []

    for workers in worker_counts:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )

        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=1e-3,)

        timer = Timer()
        model.train()
        timer.start()

        for images, labels in loader:
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

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = timer.stop()

        results.append({
            "workers": workers,
            "epoch_time": elapsed,
        })

        print(
            f"Workers={workers} "
            f"Time={elapsed:.2f}s"
        )
    return results
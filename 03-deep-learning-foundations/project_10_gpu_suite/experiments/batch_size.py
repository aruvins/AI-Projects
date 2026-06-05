import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from models.cnn import CNN
from utils.timer import Timer
from utils.metrics import throughput


def benchmark_batch_size(dataset,batch_sizes,device):
    results = []

    for batch_size in batch_sizes:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        timer = Timer()

        model.train()
        timer.start()
        total_samples = 0

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
            total_samples += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = timer.stop()

        results.append({
            "batch_size": batch_size,
            "epoch_time": elapsed,
            "throughput": throughput(
                total_samples,
                elapsed
            )
        })

        print(
            f"Batch={batch_size} "
            f"Time={elapsed:.2f}s"
        )

    return results
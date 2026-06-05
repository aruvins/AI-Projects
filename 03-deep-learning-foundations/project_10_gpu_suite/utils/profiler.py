import torch
from torch.profiler import profile, ProfilerActivity

def profile_training_step(model, loader, criterion, optimizer, device):
    model.train()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
            if device.type == "cuda"
            else ProfilerActivity.CPU
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(
            outputs,
            labels,
        )
        loss.backward()
        optimizer.step()

    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20,
        )
    )
    return prof

def get_activities(device):
    activities = [
        ProfilerActivity.CPU
    ]
    if device.type == "cuda":
        activities.append(
            ProfilerActivity.CUDA
        )
    return activities
import torch


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()

    return correct / len(labels)

def throughput(num_samples, elapsed_time):
    return num_samples / elapsed_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 ** 2)

def gpu_memory_allocated():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def gpu_memory_reserved():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 2)
    return 0.0

def benchmark_summary(model, epoch_time, samples):
    return {
        "parameters": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "epoch_time": epoch_time,
        "throughput": throughput(
            samples,
            epoch_time
        ),
    }
import time
import numpy as np
import torch


# -----------------------------
# CPU Benchmark (NumPy)
# -----------------------------

def cpu_matmul(n):
    a = np.random.randn(n, n)
    b = np.random.randn(n, n)

    start = time.time()
    _ = np.dot(a, b)
    end = time.time()

    return end - start


# -----------------------------
# GPU Benchmark (PyTorch)
# -----------------------------

def gpu_matmul(n):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        return None
    
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # warmup
    _ = torch.matmul(a, b) # matrix multiplication
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    start = time.time()
    _ = torch.matmul(a, b) # matrix multiplication
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    end = time.time()

    return end - start

# -----------------------------
# Run Benchmarks
# -----------------------------

def run_benchmarks():
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

    results = {
        "size": [],
        "cpu_time": [],
        "gpu_time": [],
    }

    for n in sizes:
        cpu_time = cpu_matmul(n)

        gpu_time = gpu_matmul(n)

        results["size"].append(n)
        results["cpu_time"].append(cpu_time)
        results["gpu_time"].append(gpu_time)

        print(f"n={n} | CPU={cpu_time:.4f}s | GPU={gpu_time}")
    return results
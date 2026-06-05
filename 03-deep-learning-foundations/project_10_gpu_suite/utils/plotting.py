import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("./outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_cpu_gpu(cpu_time, gpu_time):
    plt.figure(figsize=(6, 4))
    plt.bar(["CPU", "GPU"],[cpu_time, gpu_time])
    plt.ylabel("Epoch Time (s)")
    plt.title("CPU vs GPU Training Time")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cpu_vs_gpu.png")
    plt.show()

def plot_batch_scaling(batch_sizes, throughputs):
    plt.figure(figsize=(7, 4))
    plt.plot(batch_sizes, throughputs, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Samples / Second")
    plt.title("Batch Size Scaling")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batch_scaling.png")
    plt.show()

def plot_workers(workers, epoch_times):
    plt.figure(figsize=(7, 4))
    plt.plot(workers, epoch_times, marker="o")
    plt.xlabel("Workers")
    plt.ylabel("Epoch Time (s)")
    plt.title("DataLoader Worker Benchmark")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "workers.png")
    plt.show()

def plot_amp(fp32_time, amp_time):
    plt.figure(figsize=(6, 4))
    plt.bar(["FP32", "AMP"],[fp32_time, amp_time])
    plt.ylabel("Epoch Time (s)")
    plt.title("Mixed Precision Benchmark")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amp.png")
    plt.show()

def plot_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    print(df.head())
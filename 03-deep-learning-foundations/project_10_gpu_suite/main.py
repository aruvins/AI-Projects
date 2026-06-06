import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from experiments.cpu_vs_gpu import benchmark_device
from experiments.batch_size import benchmark_batch_size
from experiments.dataloader_workers import benchmark_workers
from experiments.mixed_precision import (
    benchmark_fp32,
    benchmark_amp,
)

from utils.benchmark import BenchmarkLogger

from utils.plotting import (
    plot_cpu_gpu,
    plot_batch_scaling,
    plot_workers,
    plot_amp,
)


def get_device():
    """
    Select the best available device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset():
    """
    Load CIFAR-10 training set.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    return dataset


def run_cpu_vs_gpu(dataset, logger):
    """
    Compare CPU and GPU training performance.
    """

    print("\n" + "=" * 50)
    print("CPU vs GPU Benchmark")
    print("=" * 50)

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    cpu_result = benchmark_device(
        torch.device("cpu"),
        loader,
    )

    logger.log(cpu_result)

    print(cpu_result)

    gpu_result = None

    if torch.cuda.is_available():

        gpu_result = benchmark_device(
            torch.device("cuda"),
            loader,
        )

        logger.log(gpu_result)

        print(gpu_result)

        plot_cpu_gpu(
            cpu_result["epoch_time"],
            gpu_result["epoch_time"],
        )

    elif torch.backends.mps.is_available():

        gpu_result = benchmark_device(
            torch.device("mps"),
            loader,
        )

        logger.log(gpu_result)

        print(gpu_result)

        plot_cpu_gpu(
            cpu_result["epoch_time"],
            gpu_result["epoch_time"],
        )


def run_batch_size_benchmark(
    dataset,
    device,
    logger,
):
    """
    Measure throughput scaling with batch size.
    """

    print("\n" + "=" * 50)
    print("Batch Size Benchmark")
    print("=" * 50)

    batch_sizes = [
        16,
        32,
        64,
        128,
        256,
        512,
    ]

    results = benchmark_batch_size(
        dataset=dataset,
        batch_sizes=batch_sizes,
        device=device,
    )

    for result in results:
        logger.log(result)
        print(result)

    plot_batch_scaling(
        [r["batch_size"] for r in results],
        [r["throughput"] for r in results],
    )


def run_worker_benchmark(
    dataset,
    device,
    logger,
):
    """
    Measure DataLoader worker impact.
    """

    print("\n" + "=" * 50)
    print("DataLoader Worker Benchmark")
    print("=" * 50)

    worker_counts = [
        0,
        2,
        4,
        8,
    ]

    results = benchmark_workers(
        dataset=dataset,
        worker_counts=worker_counts,
        batch_size=128,
        device=device,
    )

    for result in results:
        logger.log(result)
        print(result)

    plot_workers(
        [r["workers"] for r in results],
        [r["epoch_time"] for r in results],
    )


def run_mixed_precision_benchmark(
    dataset,
    logger,
):
    """
    Compare FP32 vs AMP.
    CUDA only.
    """

    if not torch.cuda.is_available():
        print(
            "\nSkipping mixed precision benchmark "
            "(CUDA unavailable)"
        )
        return

    print("\n" + "=" * 50)
    print("Mixed Precision Benchmark")
    print("=" * 50)

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
    )

    fp32_result = benchmark_fp32(
        loader,
        torch.device("cuda"),
    )

    amp_result = benchmark_amp(
        loader,
        torch.device("cuda"),
    )

    logger.log(fp32_result)
    logger.log(amp_result)

    print(fp32_result)
    print(amp_result)

    plot_amp(
        fp32_result["epoch_time"],
        amp_result["epoch_time"],
    )


def main():

    print("\nLoading CIFAR-10...\n")
    dataset = load_dataset()
    logger = BenchmarkLogger()
    device = get_device()
    print(f"Using device: {device}")

    run_cpu_vs_gpu(
        dataset,
        logger,
    )

    run_batch_size_benchmark(
        dataset,
        device,
        logger,
    )

    run_worker_benchmark(
        dataset,
        device,
        logger,
    )

    print("\n" + "=" * 50)
    print("NOTE ON MACOS RESULTS")
    print("=" * 50)

    print(
        """
    On most Linux and CUDA systems, increasing DataLoader workers
    typically reduces training time because multiple processes can
    prepare future batches while the GPU trains on the current batch.

    Expected behavior:

    Workers=0 -> Slowest
    Workers=2 -> Faster
    Workers=4 -> Faster
    Workers=8 -> Saturates

    However, on macOS (especially Apple Silicon), the opposite can occur.

    Reasons:

    1. Worker Startup Overhead
    macOS uses the 'spawn' multiprocessing method, which creates
    entirely new Python processes. Starting these workers can be
    expensive and may dominate benchmark runtime.

    2. Small Dataset Size
    CIFAR-10 images are only 32x32 pixels and require very little
    preprocessing. The cost of loading each sample is often smaller
    than the cost of managing additional worker processes.

    3. Inter-Process Communication
    Data must be transferred between worker processes and the main
    training process. This introduces additional overhead.

    4. MPS Training Characteristics
    Apple's Metal Performance Shaders (MPS) backend often benefits
    less from aggressive DataLoader parallelism than CUDA systems.

    As a result, results such as:

    Workers=0 -> 1.7s
    Workers=2 -> 14.5s
    Workers=4 -> 26.7s

    can be completely valid on macOS and demonstrate that adding more
    workers is not always beneficial. Performance optimization is highly
    dependent on the hardware platform, dataset size, and preprocessing
    pipeline.
    """
    )

    run_mixed_precision_benchmark(
        dataset,
        logger,
    )

    print("\n" + "=" * 50)
    print("Benchmark Summary")
    print("=" * 50)
    logger.summary()
    logger.save_csv(
        "benchmark_results.csv"
    )
    print("\nDone.\n")


if __name__ == "__main__":
    main()
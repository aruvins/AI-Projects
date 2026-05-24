import matplotlib.pyplot as plt

def plot_results(results):

    sizes = results["size"]
    cpu = results["cpu_time"]
    gpu = results["gpu_time"]

    plt.figure(figsize=(8, 5))

    plt.plot(sizes, cpu, label="CPU (NumPy)")

    if gpu[0] is not None:
        plt.plot(sizes, gpu, label="GPU (PyTorch CUDA)")

    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Time (seconds)")
    plt.title("CPU vs GPU Matrix Multiplication Performance")
    plt.legend()
    plt.grid(True)

    plt.savefig("output/performance.png")
    plt.close()
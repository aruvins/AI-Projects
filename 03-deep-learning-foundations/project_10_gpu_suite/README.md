# 🔴 Project 10 — GPU Training Benchmark & Optimization Suite

### 🎯 Goal
Understand performance scaling in deep learning.

### Concepts
- GPU acceleration
- batching effects
- training speed comparison

### Outcome
Benchmark CPU vs GPU training performance.

---

# GPU Training Benchmark & Optimization Suite

A deep learning systems project that investigates how hardware, batching strategies, data pipelines, and numerical precision affect neural network training performance.

This project shifts the focus from model accuracy to computational efficiency. While many machine learning projects emphasize building better models, modern AI systems are often constrained by training speed, memory bandwidth, and hardware utilization. Understanding these factors is essential for training larger models and conducting experiments efficiently.

---

# Project Structure

```text
project_10_gpu_benchmark/
│
├── experiments/
│   ├── cpu_vs_gpu.py
│   ├── batch_size.py
│   ├── dataloader_workers.py
│   └── mixed_precision.py
│
├── models/
│   ├── cnn.py
│   └── resnet18.py
│
├── utils/
│   ├── benchmark.py
│   ├── profiler.py
│   ├── metrics.py
│   ├── plotting.py
│   └── timer.py
│
├── outputs/
│   ├── charts/
│   └── logs/
│
├── main.py
├── requirements.txt
└── README.md
```

---


# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_10_gpu_suite

```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate

```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Run the Project

```bash
python main.py
```

---

# Overview

Training a neural network requires millions or even billions of mathematical operations. As models and datasets grow, training performance becomes a critical engineering challenge.

This project explores several of the most important factors that influence training speed:

* CPU vs GPU computation
* Batch size scaling
* Data loading bottlenecks
* Mixed precision training
* Throughput optimization
* Hardware utilization

Using PyTorch and CIFAR-10, the project benchmarks different training configurations and visualizes their impact on performance.

---

# Why Performance Matters

When studying machine learning, it is easy to focus entirely on accuracy.

For example:

```text
Model A Accuracy: 92%
Model B Accuracy: 93%
```

However, in production environments engineers often ask different questions:

```text
How long does training take?

How much memory does it use?

Can we train larger models?

Can we iterate faster?
```

A model that trains twice as fast can dramatically improve research productivity by allowing more experiments to be run in the same amount of time.

As AI systems scale to billions of parameters, computational efficiency becomes just as important as model quality.

---

# Deep Learning Training Pipeline

Training a neural network involves several stages:

```text
Dataset
   ↓
DataLoader
   ↓
CPU Processing
   ↓
GPU Transfer
   ↓
Forward Pass
   ↓
Loss Computation
   ↓
Backpropagation
   ↓
Parameter Update
```

Each stage can become a bottleneck.

This project investigates where those bottlenecks occur and how they affect overall performance.

---

# CPU vs GPU Training

## Why GPUs Are Faster

Neural network training is dominated by matrix multiplication.

For example:

```text
Input Matrix × Weight Matrix
```

A CPU contains a small number of powerful cores optimized for sequential execution.

```text
CPU

Core 1
Core 2
Core 3
Core 4
...
```

A GPU contains thousands of smaller cores optimized for parallel execution.

```text
GPU

■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
Thousands of Parallel Compute Units
■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
```

Since neural network operations can be executed simultaneously, GPUs provide massive speedups.

---

## Matrix Multiplication Parallelism

Suppose we compute:

```text
A × B
```

Every output element can be computed independently.

```text
Output[0,0]
Output[0,1]
Output[0,2]
...
```

A GPU distributes these calculations across thousands of threads.

This is one of the primary reasons GPUs dominate deep learning workloads.

---

## Expected Results

Typical performance:

```text
CPU
Epoch Time: 35 seconds

GPU
Epoch Time: 2 seconds
```

Actual values depend on hardware, but speedups of 10–50x are common.

---

# Batch Size Scaling

## What Is a Batch?

Instead of processing one training example at a time, neural networks process groups of examples called batches.

Example:

```text
Batch Size = 1

Image
↓
Model
↓
Prediction
```

Versus:

```text
Batch Size = 128

128 Images
↓
Model
↓
128 Predictions
```

---

## Why Larger Batches Improve Performance

GPUs are designed for large parallel workloads.

Small batches:

```text
GPU Utilization = Low
```

Many GPU cores sit idle.

Large batches:

```text
GPU Utilization = High
```

More computations are performed simultaneously.

This increases throughput:

```text
Samples Processed Per Second
```

---

## Throughput

A common performance metric is throughput.

```text
Throughput
=
Samples Processed
------------------
Training Time
```

Higher throughput indicates better hardware utilization.

---

## Memory Constraints

Increasing batch size indefinitely is not possible.

Each sample requires memory for:

* Activations
* Gradients
* Parameters

Eventually:

```text
GPU Out Of Memory
```

occurs.

The optimal batch size balances:

* Speed
* Memory usage
* Numerical stability

---

# Data Loading Bottlenecks

## The Hidden Problem

Many beginners assume the GPU is always the bottleneck.

Often the GPU spends time waiting for data.

```text
GPU Waiting...
```

This results in poor utilization.

---

## DataLoader Workers

PyTorch uses worker processes to prepare data in parallel.

```python
DataLoader(
    dataset,
    num_workers=4
)
```

Instead of one process loading data:

```text
Worker 0
```

multiple workers can prepare batches simultaneously:

```text
Worker 0
Worker 1
Worker 2
Worker 3
```

---

## Why More Workers Can Help

While the GPU trains on one batch:

```text
Batch N
```

workers prepare:

```text
Batch N+1
```

This reduces idle time.

---

## Why More Workers Can Hurt

Workers are not free.

Each worker introduces:

* Process creation overhead
* Memory overhead
* Inter-process communication

For small datasets such as CIFAR-10, excessive workers may actually reduce performance.

This is particularly common on macOS systems.

---

### Note on macOS and Apple Silicon Performance

In many deep learning workloads, increasing the number of `DataLoader` workers improves performance because data loading and preprocessing can occur in parallel with model training. As a result, the GPU spends less time waiting for new batches and more time performing useful computation.

However, on macOS systems—particularly Apple Silicon devices (M1, M2, M3, and M4)—the opposite behavior is often observed when working with small datasets such as CIFAR-10.

For example:

```text
Workers=0  -> 1.75s
Workers=2  -> 14.48s
Workers=4  -> 26.66s
```

This occurs because PyTorch uses Python multiprocessing to create worker processes. On macOS, worker processes are launched using the **spawn** method, which introduces significantly more overhead than the **fork** method commonly used on Linux systems.

When `num_workers > 0`, PyTorch must:

1. Create additional worker processes
2. Initialize the dataset in each process
3. Establish inter-process communication channels
4. Serialize and transfer data between processes

For large datasets with expensive preprocessing pipelines, this overhead is usually outweighed by the benefits of parallel data loading.

However, CIFAR-10 consists of small 32×32 images and uses minimal preprocessing:

```python
transforms.ToTensor()
```

Because loading each sample is extremely inexpensive, the cost of managing additional worker processes can exceed the cost of loading the data itself.

As a result:

```text
More workers
≠
Faster training
```

on some macOS configurations.

This benchmark demonstrates an important performance engineering principle:

> Optimization techniques are workload-dependent. A strategy that improves performance on one system may reduce performance on another.

For larger datasets, more complex augmentations, or Linux-based CUDA systems, increasing `num_workers` often improves throughput. For lightweight workloads on Apple Silicon, `num_workers=0` or `num_workers=1` may provide the best performance.

---

# Mixed Precision Training

## Floating Point Numbers

Neural networks typically use:

```text
FP32
```

which stores each value using 32 bits.

Example:

```text
0.123456789
```

---

## FP16

Mixed precision uses:

```text
FP16
```

for many computations.

Benefits:

* Less memory usage
* Faster arithmetic
* Higher throughput

---

## Why It Works

Modern GPUs contain specialized hardware called Tensor Cores.

Tensor Cores are optimized for lower precision operations.

Instead of:

```text
FP32 × FP32
```

they can efficiently compute:

```text
FP16 × FP16
```

at significantly higher speeds.

---

# Mixed Precision Training

## Floating Point Numbers in Deep Learning

Neural networks represent nearly everything as floating-point numbers:

* Input features
* Activations
* Weights
* Gradients
* Loss values

The most common format is **FP32 (32-bit floating point)**.

Each value occupies 32 bits of memory:

```text
FP32

[ Sign | Exponent | Mantissa ]

1 bit
8 bits
23 bits
```

The exponent controls the range of numbers that can be represented, while the mantissa controls numerical precision.

For example:

```text
0.123456789
```

can be stored with relatively high precision.

For decades, FP32 was the standard numerical format used for training neural networks because it provides a good balance between accuracy and stability.

---

## The Cost of FP32

The problem is that modern neural networks perform enormous numbers of calculations.

Consider a single linear layer:

```text
Input Matrix × Weight Matrix
```

If the input has shape:

```text
4096 × 4096
```

then millions of multiplications and additions are required.

Large language models and modern vision models perform billions or even trillions of floating-point operations during training.

Using FP32 means:

```text
4 bytes per value
```

for every:

* Weight
* Activation
* Gradient
* Optimizer state

As models become larger, memory consumption quickly becomes a major bottleneck.

---

## Enter FP16

FP16 (16-bit floating point) stores each value using half as many bits.

```text
FP16

[ Sign | Exponent | Mantissa ]

1 bit
5 bits
10 bits
```

Compared to FP32:

```text
FP32 = 32 bits
FP16 = 16 bits
```

This immediately reduces memory usage by roughly:

```text
50%
```

for tensors stored in FP16.

---

## Why Less Memory Matters

Memory bandwidth is one of the biggest limitations in deep learning.

Before a GPU can perform a computation, it must load data from memory.

For example:

```text
GPU Memory
      ↓
Tensor Loaded
      ↓
Matrix Multiply
      ↓
Result Stored
```

If each value is half the size:

```text
FP16 = 2 bytes
FP32 = 4 bytes
```

the GPU can move approximately twice as many values through memory at once.

This often improves training speed even before considering specialized hardware.

---

## Tensor Cores

Modern NVIDIA GPUs contain specialized hardware units called **Tensor Cores**.

Tensor Cores were introduced specifically for deep learning workloads.

Traditional GPU cores perform arithmetic operations one element at a time.

```text
a × b
c × d
e × f
...
```

Tensor Cores instead operate on small matrices simultaneously.

Conceptually:

```text
A × B
```

where:

```text
A = 16×16 matrix
B = 16×16 matrix
```

can be executed directly in hardware.

This dramatically increases throughput for matrix multiplication, which is the dominant operation in neural network training.

---

## Why Tensor Cores Prefer Lower Precision

Tensor Cores are optimized for formats such as:

```text
FP16 (Floating Point 16 [1 sign bit | 5 exponent bits | 10 mantissa bits])
```

```text
BF16 (Brain Floating 16 [1 sign bit | 8 exponent bits | 7 mantissa bits])
```

```text
TF32 (TensorFloat 32 = FP32 storage + Tensor Core execution)
```

```text
FP8 (Floating Point 8)
```

rather than traditional FP32.

Because the numbers occupy fewer bits:

```text
FP16 = 16 bits
FP32 = 32 bits
```

the hardware can process more values simultaneously.

A simplified intuition:

```text
FP32

□□□□

FP16

□□ □□
```

The same hardware resources can process more FP16 values in parallel.

This is one reason why mixed precision training often produces speedups of:

```text
1.5× – 3×
```

depending on the GPU architecture.

---

## The Numerical Stability Problem

If FP16 is so much faster, why not train everything in FP16?

Because FP16 has significantly lower precision.

Consider:

```text
FP32
0.000000123456

FP16
0.000000123456
```

Very small values may be rounded away entirely.

This becomes problematic during backpropagation.

Neural networks frequently produce extremely small gradients:

```text
0.000001
0.0000001
0.00000001
```

If these values become too small, FP16 may round them to:

```text
0
```

This phenomenon is called:

```text
Gradient Underflow
```

When gradients become zero, learning stops.

---

## Mixed Precision Training

To solve this problem, modern training systems use **mixed precision**.

The idea is simple:

Use FP16 where it is safe and FP32 where precision is important.

Instead of:

```text
Everything = FP32
```

or

```text
Everything = FP16
```

we combine both.

Typical workflow:

```text
Forward Pass
      ↓
FP16

Activation Storage
      ↓
FP16

Matrix Multiplications
      ↓
FP16 Tensor Cores

Gradient Accumulation
      ↓
FP32

Master Weights
      ↓
FP32
```

This preserves numerical stability while still gaining most of the speed benefits.

---

## Loss Scaling

Mixed precision training introduces another technique called **loss scaling**.

Suppose a gradient equals:

```text
0.0000001
```

In FP16 this may underflow to zero.

Instead, we temporarily multiply the loss by a large constant:

```text
Loss × 1024
```

which produces larger gradients:

```text
0.0001024
```

These gradients can now be represented safely.

After backpropagation, the gradients are scaled back down before updating the weights.

PyTorch's AMP system performs this automatically using:

```python
scaler = torch.cuda.amp.GradScaler()
```

---

## Automatic Mixed Precision (AMP)

PyTorch provides AMP to manage mixed precision automatically.

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    outputs = model(images)
    loss = criterion(outputs, labels)
```

AMP determines:

* Which operations should use FP16
* Which operations should remain FP32
* When loss scaling is required

This allows developers to benefit from mixed precision without manually converting tensors.

---

## Memory Savings

Mixed precision reduces memory usage in several areas:

### Activations

```text
FP32 → 4 bytes
FP16 → 2 bytes
```

### Gradients

```text
FP32 → 4 bytes
FP16 → 2 bytes
```

### Intermediate Tensors

```text
FP32 → FP16
```

The result is often:

```text
30–50% lower memory usage
```

during training.

This can allow:

* Larger batch sizes
* Larger models
* Higher resolution inputs

to fit into GPU memory.

---

## Why Mixed Precision Became Standard

Modern AI models are extremely large.

Examples include:

* GPT-style language models
* Vision Transformers
* Diffusion models

Without mixed precision, many of these models would require significantly more memory and training time.

Today, mixed precision is considered a standard optimization technique in deep learning and is used throughout industry and research.

For many workloads, enabling AMP is one of the simplest ways to achieve:

* Faster training
* Lower memory consumption
* Better hardware utilization

with minimal code changes.

In practice, most modern deep learning systems train using some form of mixed precision rather than pure FP32 arithmetic.

---

# Profiling Neural Networks

Performance optimization begins with measurement.

PyTorch includes a profiler that records:

* CPU operations
* GPU operations
* Memory allocations
* Execution times

Example output:

```text
Operation          Time

Conv2D             45%
MatMul             32%
BatchNorm           8%
ReLU                5%
```

This allows engineers to identify bottlenecks and optimize the most expensive operations.

---

# Models

## CNN Baseline

The project includes a lightweight Convolutional Neural Network.

CNNs exploit spatial structure in images through:

* Convolutions
* Weight sharing
* Local receptive fields

These properties make them highly efficient for image classification.

---

## ResNet18

The project also includes ResNet18.

ResNet introduced residual connections:

```text
Output = F(x) + x
```

which allow gradients to flow more effectively through deep networks.

Residual learning enabled the training of significantly deeper neural networks and remains one of the most influential innovations in computer vision.

---

# Benchmark Outputs

The project automatically generates:

## Charts

```text
cpu_vs_gpu.png
batch_scaling.png
workers.png
amp.png
```

## Logs

```text
benchmark_results.csv
```

These outputs make it easy to compare configurations and identify performance trends.

---

# Skills Developed

This project develops practical machine learning engineering skills:

### Deep Learning Systems

* GPU acceleration
* Parallel computation
* Memory management
* Hardware-aware optimization

### PyTorch

* DataLoaders
* Mixed precision training
* Profiling tools
* Benchmarking workflows

### Performance Engineering

* Throughput analysis
* Bottleneck identification
* Resource utilization
* Experimental methodology

---

# Key Takeaways

Modern AI is not only about building better models.

It is also about using computational resources efficiently.

Understanding how hardware, memory, parallelism, and data pipelines interact is essential for training increasingly large neural networks.

This project provides hands-on experience with the same performance considerations faced by machine learning engineers working on state-of-the-art deep learning systems.

---

# References

## Deep Learning

Goodfellow, Bengio, Courville

Deep Learning

https://www.deeplearningbook.org

---

## PyTorch Documentation

https://pytorch.org/docs/stable

---

## Automatic Mixed Precision

https://pytorch.org/docs/stable/amp.html

---

## PyTorch Profiler

https://pytorch.org/docs/stable/profiler.html

---

## CIFAR-10 Dataset

https://www.cs.toronto.edu/~kriz/cifar.html

---

## ResNet Paper

He et al. (2015)

Deep Residual Learning for Image Recognition

https://arxiv.org/abs/1512.03385

---

## NVIDIA Mixed Precision Training

Micikevicius et al. (2017)

Mixed Precision Training

https://arxiv.org/abs/1710.03740

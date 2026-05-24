# GPU & Performance Benchmark Suite

## Project 5 — Understanding Why GPUs Power Modern AI

This project benchmarks CPU vs GPU performance on matrix multiplication to reveal why GPUs are essential for deep learning.

---

# Hardware Note (MacBook Pro Compatibility)

This benchmark was run on a MacBook Pro, which does not support NVIDIA CUDA acceleration. As a result, GPU computations were executed using Apple’s Metal Performance Shaders (MPS) backend instead of CUDA.

CUDA is NVIDIA’s parallel computing platform designed specifically for accelerating deep learning workloads on NVIDIA GPUs. In contrast, MPS is Apple’s GPU acceleration framework that enables similar parallel computation on Apple Silicon devices (M1/M2/M3), allowing PyTorch to leverage the Mac’s integrated GPU for machine learning tasks.

While both CUDA and MPS provide GPU acceleration, CUDA is more widely used in large-scale AI research and production environments due to NVIDIA’s dominance in deep learning hardware and ecosystem support.

---

# Why This Project Exists

Modern AI systems like neural networks, transformers, and diffusion models rely almost entirely on one operation:

> **large-scale matrix multiplication**

Even though AI feels complex, at the hardware level it is mostly:

* multiplying matrices
* adding numbers
* repeating this billions of times

This project helps you *see* why hardware matters by comparing:

* CPU performance (NumPy)
* GPU performance (PyTorch MPS)

---

# What You Will Learn

By completing this project, you will understand:

* how CPUs process instructions
* how GPUs process data in parallel
* why deep learning is compute-heavy
* why batching improves performance
* how matrix size affects computation time
* why GPUs dominate AI workloads

---

# Core Concept: Matrix Multiplication in AI

Almost every AI model relies on this operation:

```text
output = input × weights
```

This appears in:

* neural network layers
* attention mechanisms (Transformers)
* embeddings
* convolution operations

So if we understand matrix multiplication speed, we understand AI performance.

---

# CPU vs GPU: The Fundamental Difference

## 1. CPU (Central Processing Unit)

A CPU is designed to be:

> fast at doing a few complex tasks at once

### Key characteristics:

* 4–16 strong cores (typical consumer CPU)
* very high clock speed
* optimized for sequential logic
* excellent at decision-making and branching

### CPU analogy:

A CPU is like:

> a few very smart workers who can solve hard problems one at a time

### Strengths:

* complex logic (if statements, loops)
* general-purpose computing
* operating systems, apps, browsers

### Weakness in AI:

AI requires:

* millions of identical operations
* repeated matrix calculations

CPUs are not optimized for this scale.

---

## 2. GPU (Graphics Processing Unit)

A GPU is designed for:

> doing thousands of simple operations at the same time

### Key characteristics:

* hundreds to thousands of small cores
* lower clock speed than CPU
* extremely high parallel throughput
* optimized for mathematical operations

### GPU analogy:

A GPU is like:

> a massive factory with thousands of workers doing the same task simultaneously

---

# Why GPUs Are Faster for AI

AI workloads are:

* repetitive
* structured
* mathematically identical operations

Example: multiplying two matrices

Each element in the output matrix is independent:

```text
C[i][j] = sum(A[i][k] * B[k][j])
```

This means:

> each computation can be done in parallel

---

## CPU Execution

A CPU might compute:

* one row at a time
* or a few elements at once

So computation is partially sequential.

---

## GPU Execution

A GPU computes:

* thousands of matrix elements simultaneously

So instead of:

```text
1 → 2 → 3 → 4 → 5
```

it does:

```text
1 2 3 4 5 ... all at once
```

---

# The Key Insight

> AI performance is not limited by intelligence, but by parallel computation capacity.

---

# What This Project Measures

This benchmark compares CPU and GPU performance on:

* square matrix multiplication
* increasing sizes:

  * 128 × 128
  * 256 × 256
  * 512 × 512
  * 1024 × 1024
  * 2048 × 2048
  * 4096 x 4096
  * 8192 x 8192

Each test measures:

```text
time to compute A × B
```

---

# Expected Results

## CPU Behavior

* slower growth as matrix size increases
* becomes inefficient at large matrices
* struggles with scaling

## GPU Behavior

* significantly faster for large matrices
* improves dramatically with size
* handles large workloads efficiently

## Important Observation

At small sizes:

* CPU may appear faster (GPU overhead)

At large sizes:

* GPU dominates significantly

---

# Project Structure

```text
gpu_benchmark_suite/
│
├── main.py              # Runs benchmarks
├── benchmark.py         # CPU and GPU timing logic
├── plotting.py          # Visualization
├── requirements.txt
└── output/
    └── performance.png
```

---
# How to Run

## 1. change directory into project 3

### macOS/Linux

```bash
cd 01-python-foundations/project_5_gpu_performance/
```

## 2. Create Virtual Environment

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Benchmark

```bash
python main.py
```

---

## 3. Output

The program generates:

```text
output/performance.png
```

This plot shows CPU vs GPU execution time across matrix sizes.

---

# Deep Learning Connection

This project directly explains why GPUs dominate AI:

Neural networks require:

* forward propagation → matrix multiplication
* backpropagation → matrix multiplication
* gradient updates → matrix multiplication

So training a model involves:

> millions of matrix operations per second

---

# What is batching?

Instead of feeding a neural network **one data sample at a time**, we feed it **a group of samples together**.

That group is called a **batch**.

So instead of:

```text id="b1"
1 image → forward pass → update weights
1 image → forward pass → update weights
1 image → forward pass → update weights
```

we do:

```text id="b2"
32 images → forward pass → compute loss → update weights once
```

---

# Why not just use one sample at a time?

You technically *can*, but it is extremely inefficient.

If you train one image at a time:

* the GPU sits idle most of the time
* computation is not parallelized
* training is noisy and unstable

This is called **stochastic gradient descent (batch size = 1)**.

---

# What batching actually does (step by step)

Let’s say your batch size is 32.

### Step 1 — Load data

You take 32 samples:

```text id="b3"
X_batch = [x1, x2, ..., x32]
```

---

### Step 2 — Forward pass (in parallel)

Instead of processing one input vector:

```text id="b4"
output = model(x1)
```

you process the entire batch at once:

```text id="b5"
outputs = model(X_batch)
```

Behind the scenes, this becomes matrix operations like:

```text id="b6"
(32 × input_size) × (input_size × hidden_size)
```

This is where GPUs shine.

---

### Step 3 — Compute loss

You compute loss for all 32 predictions:

```text id="b7"
loss = average(loss(x1), loss(x2), ..., loss(x32))
```

So instead of one error signal, you get a **more stable estimate of error**.

---

### Step 4 — Backpropagation

Now gradients are computed using all 32 samples at once.

This gives:

* smoother gradients
* less noise
* more stable learning

---

### Step 5 — Update weights once

Instead of updating after every single sample:

```text id="b8"
update weights 32 times
```

you do:

```text id="b9"
update weights 1 time (based on batch)
```

---

# Why batching is faster (the key insight)

GPUs are designed for **parallel math**, not sequential work.

A GPU has:

* thousands of small cores
* optimized matrix multiplication hardware

So when you give it a batch:

```text id="b10"
32 samples → one big matrix operation
```

it can compute everything simultaneously.

---

# Intuition: CPU vs GPU batching

## CPU (bad at batching)

A CPU is like:

> 4–16 very smart workers doing tasks one after another

So batching doesn’t help much.

---

## GPU (perfect for batching)

A GPU is like:

> 10,000 workers doing the same operation at the same time

So batching turns:

```text id="b11"
32 small tasks
```

into:

```text id="b12"
1 large parallel task
```

---

# Why batching improves learning quality

Batching doesn’t just make things faster — it also improves training stability.

### Without batching (batch size = 1):

* very noisy updates
* model “jumps around”
* unstable convergence

### With batching (32–128):

* smoother gradient estimates
* more stable learning
* better final accuracy

---

# Tradeoff: batch size matters

Bigger batch size:

✔ faster GPU utilization
✔ smoother gradients
❌ more memory usage
❌ can reduce generalization sometimes

Smaller batch size:

✔ more randomness (can help generalization)
✔ less memory
❌ slower training

---

# One-sentence summary

> Batching is the process of grouping multiple training samples together so that neural networks can process them in parallel, making GPU computation fast and efficient while also stabilizing learning.

---

# Important Insight

The breakthrough that enabled modern AI was not just algorithms.

It was hardware.

> Deep learning became possible because GPUs made large-scale matrix computation fast enough to be practical.

---

# Key Takeaways

After this project, you should understand:

* CPUs are optimized for sequential logic
* GPUs are optimized for parallel math
* AI workloads are massively parallel
* matrix multiplication is the core bottleneck
* GPU speedup grows with problem size
* batching improves efficiency

---

# Final Conceptual Summary

If you remember only one idea:

> AI is just math — and GPUs are machines built to do that math extremely fast in parallel.
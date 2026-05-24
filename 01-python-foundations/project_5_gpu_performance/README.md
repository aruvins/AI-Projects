# GPU & Performance Benchmark Suite

## Project 5 — Understanding Why GPUs Power Modern AI

This project benchmarks CPU vs GPU performance on matrix multiplication to reveal why GPUs are essential for deep learning.

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
* GPU performance (PyTorch CUDA)

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

# Why Batching Matters

AI frameworks process data in batches:

Instead of:

```text
1 sample at a time
```

they process:

```text
32 / 64 / 128 samples at once
```

This allows GPUs to maximize parallel usage.

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
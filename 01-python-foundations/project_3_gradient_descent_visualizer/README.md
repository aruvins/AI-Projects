# Gradient Descent Visualizer (Project 3)

A beginner-friendly AI project that teaches the core idea behind how neural networks learn.

This project visualizes:

- gradient descent
- optimization
- loss minimization
- convergence
- divergence
- learning rates

By the end of this project, you will understand the fundamental algorithm used to train nearly all modern AI systems.

---

# Why This Project Matters

Every machine learning model learns by adjusting parameters.

The model starts with random guesses.
Then it measures how wrong it is.
Then it slightly changes its parameters to reduce the error.

This process repeats thousands or millions of times.

That process is called:

## Gradient Descent

Gradient descent is the engine behind:

- neural networks
- deep learning
- logistic regression
- transformers
- computer vision
- large language models

Understanding this algorithm is one of the most important steps in learning AI.

---

# The Big Idea

Imagine standing on a mountain in thick fog.

You want to reach the bottom.

You cannot see the entire mountain.
You can only feel the slope beneath your feet.

So you:

1. look at the slope
2. step downhill
3. repeat

Eventually you reach the bottom.

That is exactly how gradient descent works.

---

# Core AI Concepts

## 1. Parameters

Parameters are the numbers an AI model adjusts while learning.

Examples:

- weights in a neural network
- coefficients in regression
- embeddings in transformers

In this project, we use only one parameter:

```python
x
````

---

## 2. Loss Function

A loss function measures how wrong the model is.

Lower loss = better model.

Our project uses:

[
f(x) = (x - 3)^2 + 2
]

The minimum occurs at:

[
x = 3
]

The optimizer tries to move toward this minimum.

---

## 3. Derivative (Intuition)

A derivative tells us:

* which direction is uphill
* how steep the hill is

If the derivative is:

* positive → move left
* negative → move right
* zero → you reached the minimum

For our function:

[
f'(x) = 2(x - 3)
]

This gives the slope at every point.

---

## 4. Gradient Descent Update Rule

This is the core learning equation used throughout AI.

[
x = x - η · ∇f(x)
]

Where:

Where:

- `x` = parameter
- `η` = learning rate (eta)
- `∇f(x)` = gradient (a vector that points in the direction of the greatest rate of increase of a function)

In plain English:

> new parameter = old parameter − small downhill step

---

## 5. Learning Rate

The learning rate controls step size.

Small learning rate:

* slow learning
* stable convergence

Large learning rate:

* faster movement
* may overshoot
* may diverge completely

This project visualizes all three behaviors.

---

# What You Will See

The animation compares multiple learning rates.

Example:

```python
LEARNING_RATES = [0.05, 0.2, 0.9]
```

## Small Learning Rate

The parameter moves slowly toward the minimum.

## Good Learning Rate

The parameter converges quickly and smoothly.

## Large Learning Rate

The parameter overshoots and may bounce wildly.

This is one of the most important intuitions in deep learning.

---

# How to Run

## 1. change directory into project 3

### macOS/Linux

```bash
cd 01-python-foundations/project_3_gradient_descent_visualizer/
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

## 4. Run the Visualizer

```bash
python main.py
```

---

# Experiment Ideas

## Try Different Learning Rates

Edit:

```python
LEARNING_RATES = [0.001, 0.1, 1.5]
```

Observe:

* slow convergence
* fast convergence
* divergence

---

## Change Starting Position

```python
START_X = -8
```

---

## Try a More Complex Function

In `main.py`, replace:

```python
quadratic_function
quadratic_gradient
```

with:

```python
wavy_function
wavy_gradient
```

This demonstrates:

* local minima
* more difficult optimization
* non-convex landscapes

---

# Important AI Insight

Neural network training is simply:

1. make prediction
2. calculate loss
3. compute gradients
4. update parameters
5. repeat millions of times

That entire process is powered by gradient descent.

Modern deep learning is essentially very advanced gradient descent applied to enormous neural networks.

---

# Recommended Next Projects

After this project, you should learn:

1. Linear Regression from Scratch
2. Backpropagation
3. Neural Networks with NumPy
4. PyTorch Autograd
5. CNNs and Transformers

---

# Expected Learning Outcome

After completing this project, you should understand:

* what optimization means
* how AI systems learn
* what gradients represent
* why learning rates matter
* how convergence works
* why training can fail

These ideas form the foundation of modern AI.

```
```

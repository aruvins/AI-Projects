# README.md

# Neural Network From Scratch

A beginner-friendly deep learning project that builds a fully functional neural network using only NumPy.

No PyTorch.
No TensorFlow.
No high-level AI frameworks.

This project teaches how neural networks actually work internally.

---

# Why This Project Matters

Modern AI frameworks automate nearly everything.

When you train a neural network using PyTorch or TensorFlow, the framework automatically:

- computes gradients
- performs backpropagation
- updates weights
- tracks losses
- optimizes parameters

This convenience is powerful.

But it can also hide how deep learning actually works.

This project removes the abstraction.

You will manually implement:

- forward propagation
- backpropagation
- activation functions
- gradient descent
- weight updates
- loss computation

By the end, neural networks will no longer feel like magic.

---

# What Is a Neural Network?

A neural network is a mathematical function that learns patterns from data.

It consists of layers of neurons.

Each neuron:

1. receives inputs
2. multiplies them by weights
3. adds biases
4. applies an activation function
5. passes outputs forward

This process repeats layer after layer.

---

# Core Deep Learning Concepts

## 1. Forward Propagation

Forward propagation is how information flows through the network.

Input → Hidden Layer → Output Layer

The network takes input data and produces predictions.

### Dense Layer Equation

```text
output = input · weights + bias
````

This is the foundation of neural networks.

---

## 2. Activation Functions

Without activation functions, neural networks would only learn linear relationships.

Activation functions introduce nonlinearity.

This allows networks to learn:

* curves
* boundaries
* patterns
* complex relationships

### ReLU Activation

This project uses ReLU:

```text
ReLU(x) = max(0, x)
```

ReLU is one of the most common activations in modern AI.

---

## 3. Softmax

The output layer uses Softmax.

Softmax converts raw scores into probabilities.

Example:

```text
[2.1, 0.3, 1.2]
```

becomes:

```text
[0.65, 0.09, 0.26]
```

The probabilities sum to 1.

---

## 4. Loss Functions

The loss function measures how wrong the network is.

Lower loss = better predictions.

This project uses:

## Cross-Entropy Loss

Cross-entropy is the standard loss function for classification.

It heavily penalizes confident wrong predictions.

---

## 5. Backpropagation

Backpropagation is the algorithm that powers deep learning.

It computes how much each weight contributed to the error.

Then it sends gradients backward through the network.

This allows every parameter to learn.

Backpropagation uses the chain rule from calculus.

But conceptually, it is simply:

> figuring out which parameters caused mistakes

---

## 6. Gradient Descent

Once gradients are computed, the network updates weights.

### Weight Update Rule

```text
weight = weight - learning_rate × gradient
```

This moves parameters in the direction that reduces error.

Repeated thousands of times, the network gradually learns.

---

# Dataset

This project uses a synthetic spiral dataset.

Why?

Because spiral data is:

* small
* visual
* nonlinear
* difficult for linear models

This makes it perfect for understanding neural networks.

---

# What You Will Build

A neural network that:

* classifies spiral data
* computes gradients manually
* trains using backpropagation
* tracks loss and accuracy
* visualizes learning progress
* generates decision boundaries

---

# Output Visualizations

The project saves:

```text
output/
├── training_loss.png
├── training_accuracy.png
└── predictions.png
```

These visualizations help you see how learning evolves.

---
# How to Run

## 1. change directory into project 3

### macOS/Linux

```bash
cd 01-python-foundations/project_4_neural_network/
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

## 4. Run the Project

```bash
python main.py
```

---

## Project Structure

```text
neural_network_from_scratch/
│
├── main.py
├── network.py
├── layers.py
├── activations.py
├── losses.py
├── data.py
├── visualization.py
├── requirements.txt
├── output/
│   ├── training_loss.png
│   ├── training_accuracy.png
│   └── predictions.png
└── README.md
```

---

# requirements.txt

```text
numpy
matplotlib
```

---

# Expected Training Behavior

During training you should observe:

* loss decreasing
* accuracy increasing
* decision boundaries improving

This means the network is learning patterns from data.

---

# Important AI Insight

Large modern AI systems are built from the same principles used here.

GPT models, image generators, and computer vision systems all rely on:

* matrix multiplication
* activations
* gradients
* backpropagation
* optimization

The difference is scale.

This small project contains the same fundamental learning mechanics as modern deep learning.

---

# Suggested Experiments

## Try Different Learning Rates

In `main.py`:

```python
learning_rate=0.001
```

or:

```python
learning_rate=1.0
```

Observe:

* slow learning
* fast learning
* unstable learning

---

## Change Hidden Layer Size

Try:

```python
hidden_size=8
```

or:

```python
hidden_size=128
```

Larger networks can learn more complex patterns.

---

## Add More Layers

Try implementing:

* another dense layer
* sigmoid activation
* tanh activation
* dropout

This is how deeper networks are built.

---

# Recommended Next Projects

After this project, consider learning:

1. Convolutional Neural Networks (CNNs)
2. PyTorch Autograd
3. Transformers
4. Attention Mechanisms
5. Residual Networks
6. Reinforcement Learning

---

# Expected Learning Outcome

After completing this project, you should understand:

* how neural networks compute outputs
* how backpropagation works
* how gradients are calculated
* how parameters learn
* why activation functions matter
* how deep learning frameworks operate internally

You will now understand what PyTorch and TensorFlow are doing behind the scenes.

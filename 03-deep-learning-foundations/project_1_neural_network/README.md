# 🧠 Neural Network From Scratch (NumPy)

A beginner-friendly implementation of a neural network built entirely from scratch using only NumPy.

This project demonstrates how neural networks actually work internally without using deep learning frameworks like PyTorch or TensorFlow.

The network learns the XOR problem using:

* forward propagation
* backpropagation
* gradient descent
* matrix multiplication
* activation functions

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_1_neural_network

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

# 4. Run The Project

```bash
python main.py

```

---

# 🎯 Project Goal

The goal of this project is to understand the core mathematics and mechanics behind neural networks.

Instead of relying on high-level libraries, this project manually implements:

* neurons
* hidden layers
* activations
* loss computation
* weight updates
* learning through gradients

This helps build intuition for modern AI systems and deep learning architectures.

---

# 📦 Technologies Used

* Python
* NumPy
* Matplotlib

---

# 📁 Project Structure

```text
neural_network_from_scratch/
│
├── outputs/
│   ├── loss.png
│   └── predictions.png
│
├── utils/
│   └── network.py
├── main.py
└── README.md
```

---

# 🧠 What Is a Neural Network?

A neural network is a mathematical system designed to learn patterns from data.

It works by:

1. taking input data
2. transforming it through layers of neurons
3. making predictions
4. measuring error
5. adjusting internal weights to improve future predictions

The network gradually learns by repeatedly reducing its prediction error.

---

# 🧠 The XOR Problem

This project trains the network on the XOR dataset.

| x1 | x2 | y |
| -- | -- | - |
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

XOR is historically important because:

* it cannot be solved using a single neuron
* it requires a hidden layer
* it demonstrates nonlinear learning

This makes XOR one of the most famous beginner neural network problems.

---

# 🧠 Forward Propagation

Forward propagation is the process of passing data through the network to generate predictions.

The neuron computes:

```math
z = w_1x_1 + w_2x_2 + b
```

Where:

* `x` = inputs
* `w` = weights
* `b` = bias

The result is then passed through an activation function.

---

## 🧠 What Is Bias?

Bias is a learnable value added to a neuron that helps shift its activation output.

Without bias, a neuron’s output would always be forced through the origin, making the network much less flexible.

The neuron computes:

```math id="z6ru96"
z = w_1x_1 + w_2x_2 + b
```

Where:

* `w` controls how strongly inputs affect the neuron
* `b` controls the neuron’s activation threshold

Bias allows neural networks to:

* shift decision boundaries
* learn more complex patterns
* activate even when inputs are zero

In deep learning, weights detect patterns while biases help determine when neurons should activate.

---

# 🧠 Activation Functions

Activation functions introduce nonlinearity into the network.

Without activation functions, neural networks would only be capable of learning simple linear relationships.

This project uses the sigmoid activation function:

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

The sigmoid function:

* compresses values between 0 and 1
* behaves similarly to probability
* allows smooth gradient computation

---

# 🧠 Hidden Layers

The hidden layer allows the network to learn intermediate representations of the data.

For XOR, hidden neurons may learn concepts like:

* whether inputs differ
* whether one specific input is active
* combinations of input patterns

The output layer combines these learned patterns to produce final predictions.

This process is called feature learning.

---

# 🧠 Loss Function

The network needs a way to measure how wrong its predictions are.

This project uses Mean Squared Error (MSE):

```math
MSE = \frac{1}{n}\sum(y_{true} - y_{pred})^2
```

The loss function provides a numerical score representing prediction error.

Training aims to minimize this loss.

---

# 🧠 Backpropagation

Backpropagation is the algorithm that allows neural networks to learn.

After making predictions:

1. the network computes error
2. gradients are calculated
3. weights are adjusted
4. predictions improve over time

Backpropagation uses calculus and the chain rule to determine:

* how much each weight contributed to the error
* how each weight should change

This is the core learning mechanism behind deep learning.

---

# 🧠 Gradient Descent

Gradient descent is the optimization process used to reduce error.

Weights are updated using:

```math
w = w - \eta \frac{\partial L}{\partial w}
```

Where:

* `L` = loss
* `η` = learning rate
* `∂L/∂w` = gradient of the loss with respect to the weight

The gradient points in the direction of increasing error.

Subtracting the gradient moves the model toward lower error.

---

# 🧠 Why Matrix Multiplication Matters

Neural networks rely heavily on matrix operations because they allow:

* efficient computation
* parallel processing
* GPU acceleration

For example:

```python
z1 = np.dot(X, W1) + b1
```

This computes activations for an entire layer simultaneously.

Modern deep learning frameworks optimize these operations extensively.

---

# 🧠 Why Deep Learning Works

Deep learning works because multiple layers can learn increasingly complex representations.

For example:

* early layers learn simple patterns
* deeper layers combine them into higher-level concepts

In image models:

* early layers detect edges
* middle layers detect shapes
* deeper layers detect objects

This hierarchical learning is one of the key breakthroughs in AI.

---

# 📊 Visualization

The project generates visualizations in the `outputs/` folder.

## Training Loss

`loss.png`

* shows how error decreases during training
* demonstrates the learning process

## Predictions

`predictions.png`

* visualizes final XOR predictions
* shows how the network separates classes

Visualization is extremely important in machine learning because it helps debug:

* training instability
* overfitting
* poor convergence
* bad hyperparameters

---

# ✅ Example Output

```python
[[0.049]
 [0.952]
 [0.952]
 [0.054]]
```

These predictions are very close to the correct XOR outputs:

* 0
* 1
* 1
* 0

This means the network successfully learned the XOR relationship.

---

# 🚀 Future Improvements

Possible next steps:

* ReLU activation
* Softmax outputs
* Cross-entropy loss
* Multiple hidden layers
* Mini-batch training
* Momentum optimization
* Dropout regularization
* PyTorch implementation
* GPU acceleration

---

# 🧠 What This Project Teaches

This project teaches the foundations behind:

* deep learning
* transformers
* computer vision
* large language models
* reinforcement learning

Modern AI systems are built on the same core ideas:

* tensors
* gradients
* optimization
* layered representations

Understanding these fundamentals makes advanced AI concepts significantly easier to learn later.

---

# 📚 Key Takeaway

Frameworks like PyTorch and TensorFlow automate:

* gradient computation
* tensor operations
* GPU execution

But the underlying concepts are the same ones implemented manually in this project.

Building a neural network from scratch is one of the best ways to truly understand how AI works internally.

# 👕 Fashion-MNIST Classifier

A Deep Learning project that classifies clothing images from the Fashion-MNIST dataset using a Multi-Layer Perceptron (MLP) built with PyTorch.

This project introduces core neural network concepts including forward propagation, backpropagation, optimization, validation, regularization, overfitting, and model evaluation.

---

# 📂 Project Structure

```text
project_3_fashion_mnist_classifier/

│
├── data/
│
├── outputs/
│   ├── fashion_classifier.pth
│   ├── training_loss.png
│   ├── validation_accuracy.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── main.py
├── utils/
│   └── model.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_3_fashion_mnist_classifier

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

The goal of this project is to build a neural network capable of recognizing different categories of clothing from grayscale images.

Unlike the classic MNIST handwritten digit dataset, Fashion-MNIST is a more challenging computer vision dataset because many clothing items have similar visual features.

Examples include:

* Shirts vs T-Shirts
* Coats vs Pullovers
* Sneakers vs Ankle Boots

The objective is not only to achieve high accuracy, but also to learn how neural networks generalize to unseen data.

---

# 📚 Concepts Covered

This project introduces several important Deep Learning concepts:

* Neural Networks
* Forward Propagation
* Activation Functions
* Loss Functions
* Backpropagation
* Gradient Descent
* Adam Optimizer
* Training vs Validation vs Test Data
* Overfitting
* Regularization
* Dropout
* Model Evaluation
* Confusion Matrices

---

# 🖼️ About Fashion-MNIST

Fashion-MNIST is a computer vision dataset created by Zalando Research.

The dataset contains:

| Dataset  | Images |
| -------- | ------ |
| Training | 60,000 |
| Test     | 10,000 |

Each image is:

```text
28 × 28 grayscale pixels
```

with values ranging from:

```text
0 → black
255 → white
```

The model must classify each image into one of 10 categories.

| Label | Class       |
| ----- | ----------- |
| 0     | T-Shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

# 🧠 Neural Networks

A neural network is a mathematical function that learns patterns from data.

Instead of manually defining rules, we allow the model to learn useful relationships directly from examples.

The network learns by adjusting parameters called weights.

A neural network consists of:

```text
Input Layer
    ↓
Hidden Layers
    ↓
Output Layer
```

For Fashion-MNIST:

```text
784 Inputs
    ↓
256 Neurons
    ↓
128 Neurons
    ↓
10 Outputs
```

The input layer contains:

```text
28 × 28 = 784 pixels
```

Each pixel becomes a numerical feature used by the model.

---

# 🔄 Forward Propagation

Forward propagation is the process of generating predictions.

The input image passes through every layer of the network.

Each neuron computes:

```math
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
```

Where:

* x = inputs
* w = weights
* b = bias

The result is passed through an activation function before moving to the next layer.

This process continues until the model produces class scores for all clothing categories.

---

# ⚡ ReLU Activation Function

Neural networks require activation functions to learn complex patterns.

This project uses ReLU:

```math
f(x)=\max(0,x)
```

ReLU outputs:

```text
Positive values → unchanged
Negative values → 0
```

Benefits:

* Simple
* Fast
* Helps deep networks learn efficiently
* Reduces vanishing gradient problems

Without activation functions, neural networks would behave like simple linear models regardless of depth.

---

# 🎯 Output Layer

The final layer contains:

```python
nn.Linear(128, 10)
```

The network produces 10 scores:

```text
[2.1, 0.4, 1.8, 4.5, ...]
```

Each score represents the model's confidence for a clothing category.

The highest score becomes the prediction.

For example:

```text
Class 3 (Dress)
```

would be selected if it has the highest score.

---

# 🏗️ Understanding the Network Architecture

The Fashion-MNIST classifier is built using several different neural network layers. Two of the most important components are:

* `nn.Linear()`
* `nn.ReLU()`

Understanding why these layers are used helps explain how neural networks learn patterns from data.

---

# 🔹 Why Do We Use `nn.Linear()`?

A Linear layer is the most basic building block of a neural network.

In PyTorch:

```python
nn.Linear(input_features, output_features)
```

For example:

```python
nn.Linear(784, 256)
```

means:

```text
784 input features
        ↓
256 output features
```

Each output neuron is connected to every input neuron.

Because every neuron connects to every neuron in the next layer, this is often called a:

```text
Fully Connected Layer
```

or

```text
Dense Layer
```

---

## What Does a Linear Layer Actually Do?

A Linear layer performs a mathematical operation:

```math
y = Wx + b
```

Where:

* `x` = input values
* `W` = learned weights
* `b` = learned bias
* `y` = output values

The weights and biases are the parameters the model learns during training.

You can think of the Linear layer as:

```text
A feature combiner
```

It learns which input features are important and how they should be combined.

For example:

```text
Pixel A → important
Pixel B → less important
Pixel C → very important
```

The network learns these relationships automatically.

---

## Why Is `nn.Linear()` Used In This Project?

Fashion-MNIST images are relatively small:

```text
28 × 28 pixels
```

After flattening:

```text
28 × 28 = 784 features
```

An MLP (Multi-Layer Perceptron) works well because:

* images are small
* architecture is simple
* easy for beginners to understand
* introduces core neural network concepts

The Linear layers allow the network to learn patterns from the image pixels.

---

## Other Types of Layers

Linear layers are not the only option.

### 1. Convolutional Layers

```python
nn.Conv2d()
```

Used in:

* CNNs
* Computer Vision
* Image Classification

Example:

```python
nn.Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3
)
```

Advantages:

* Detect edges
* Detect textures
* Detect shapes
* Require fewer parameters

CNNs usually outperform MLPs on image tasks.

---

### 2. Recurrent Layers

```python
nn.RNN()
nn.LSTM()
nn.GRU()
```

Used for:

* Text
* Language
* Time Series
* Speech

These layers can remember previous information.

---

### 3. Transformer Layers

```python
nn.TransformerEncoder()
```

Used for:

* Large Language Models
* GPT-style systems
* Modern AI

Transformers learn relationships between all inputs simultaneously.

---

# 🔹 Why Do We Use `nn.ReLU()`?

Without activation functions, neural networks cannot learn complex patterns.

Suppose we had:

```python
nn.Linear(784, 256)
nn.Linear(256, 128)
nn.Linear(128, 10)
```

with no activation functions.

Even though there are multiple layers, the network would behave exactly like:

```python
nn.Linear(784, 10)
```

The extra layers would provide no additional learning power.

Activation functions solve this problem.

---

## What Does ReLU Do?

ReLU stands for:

```text
Rectified Linear Unit
```

It applies:

```math
f(x) = max(0, x)
```

Examples:

```text
Input  → Output

5      → 5
2      → 2
0      → 0
-3     → 0
-10    → 0
```

Negative values become zero.

Positive values remain unchanged.

---

## Why Is ReLU Popular?

ReLU became the standard activation function because it is:

### Fast

Very simple calculation:

```text
max(0, x)
```

This makes training efficient.

---

### Helps Deep Networks Learn

Older activation functions often caused:

```text
Vanishing Gradients
```

where gradients became extremely small.

When gradients become tiny:

```text
Weights stop updating
Learning slows dramatically
```

ReLU greatly reduces this problem.

---

### Produces Sparse Activations

Many neurons become:

```text
0
```

for a given input.

This often improves efficiency and generalization.

---

# Other Activation Functions

ReLU is not the only option.

---

## Sigmoid

```python
nn.Sigmoid()
```

Formula:

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

Output range:

```text
0 → 1
```

Useful for:

* Binary Classification
* Probabilities

Example:

```text
Spam vs Not Spam
Fraud vs Not Fraud
```

Problem:

```text
Vanishing Gradients
```

in deep networks.

---

## Tanh

```python
nn.Tanh()
```

Output range:

```text
-1 → 1
```

Advantages:

* Zero centered
* Stronger gradients than Sigmoid

Still suffers from vanishing gradients.

---

## Leaky ReLU

```python
nn.LeakyReLU()
```

Instead of:

```text
negative values → 0
```

it allows a small negative slope.

Example:

```text
-10 → -0.1
```

Benefits:

* Prevents dead neurons
* Often performs similarly to ReLU

---

## GELU

```python
nn.GELU()
```

Used by:

* Transformers
* BERT
* GPT models

Advantages:

* Smooth activation
* Strong performance in large models

Disadvantages:

* More computationally expensive

---

## ELU

```python
nn.ELU()
```

Another alternative to ReLU.

Benefits:

* Allows negative outputs
* Helps gradient flow

Often used in deeper architectures.

---

# Why ReLU Is The Best Choice Here

For a beginner Fashion-MNIST classifier:

```python
nn.ReLU()
```

is usually the best choice because it is:

* Simple
* Fast
* Easy to understand
* Industry standard
* Works extremely well for MLPs

More advanced projects may replace ReLU with:

```python
nn.LeakyReLU()
nn.GELU()
```

but ReLU remains the most common starting point when learning Deep Learning.

---

# Summary

| Component                 | Purpose                                               |
| ------------------------- | ----------------------------------------------------- |
| `nn.Linear()`             | Learns relationships between features                 |
| `nn.ReLU()`               | Adds non-linearity so complex patterns can be learned |
| `nn.Conv2d()`             | Alternative layer for images                          |
| `nn.LSTM()`               | Alternative layer for sequences                       |
| `nn.TransformerEncoder()` | Alternative layer for modern AI systems               |
| `nn.Sigmoid()`            | Binary classification activation                      |
| `nn.Tanh()`               | Activation with outputs from -1 to 1                  |
| `nn.LeakyReLU()`          | ReLU variant that avoids dead neurons                 |
| `nn.GELU()`               | Modern activation used in Transformers                |

Together, the Linear layers learn useful feature representations while ReLU gives the network the ability to model complex, non-linear relationships in the Fashion-MNIST dataset.

---

# 📉 Loss Function

The loss function measures prediction error.

This project uses:

```python
nn.CrossEntropyLoss()
```

Cross-Entropy compares:

```text
Predicted probabilities
vs
Correct label
```

If the prediction is wrong:

```text
Large Loss
```

If the prediction is correct:

```text
Small Loss
```

The objective of training is to minimize this loss.

---

# 🔙 Backpropagation

Backpropagation is how neural networks learn.

After computing loss:

1. Calculate prediction error
2. Compute gradients
3. Send gradients backward through the network
4. Update weights

The gradient tells the model:

```text
Which direction should each weight move?
```

Weights that contribute to errors are adjusted accordingly.

This process occurs after every training batch.

---

# ⛰️ Gradient Descent

Gradient Descent is the optimization process used to minimize loss.

Imagine standing on a mountain in dense fog.

Your objective is to reach the lowest point.

The gradient tells you:

```text
Which direction is downhill?
```

The optimizer repeatedly updates weights until the loss becomes as small as possible.

---

# 🚀 Adam Optimizer

This project uses:

```python
torch.optim.Adam()
```

Adam stands for:

```text
Adaptive Moment Estimation
```

Adam improves upon standard Gradient Descent by tracking:

### First Moment

The average direction of recent gradients.

```text
Momentum
```

This helps the model move consistently toward better solutions.

### Second Moment

The average magnitude of recent gradients.

This allows Adam to automatically adjust learning rates for individual parameters.

Benefits:

* Fast convergence
* Stable training
* Minimal tuning required
* Excellent default optimizer

Because of these advantages, Adam is one of the most widely used optimizers in modern Deep Learning.

---

# 📦 Mini-Batch Training

Training uses:

```python
batch_size = 64
```

Instead of processing one image at a time, the model processes:

```text
64 images simultaneously
```

Benefits include:

* Faster training
* Better GPU utilization
* More stable gradient estimates

Each batch produces one weight update.

---

# 🎲 Train / Validation / Test Split

The dataset is divided into multiple portions.

### Training Set

Used to learn model parameters.

```text
80%
```

### Validation Set

Used to monitor generalization.

```text
20%
```

The validation set helps identify overfitting during training.

### Test Set

Used only after training is complete.

The test set provides an unbiased estimate of real-world performance.

---

# ⚠️ Overfitting

Overfitting occurs when a model memorizes training examples instead of learning general patterns.

Symptoms:

```text
Training Accuracy ↑
Validation Accuracy ↓
```

An overfit model performs well on training data but poorly on unseen data.

The goal of Deep Learning is not memorization.

The goal is generalization.

---

# 🛡️ Regularization

Regularization techniques help prevent overfitting.

This project uses Dropout regularization.

Regularization encourages the model to learn robust patterns that transfer to new data.

---

# 🎲 Dropout

The model includes:

```python
nn.Dropout(0.3)
```

During training:

```text
30% of neurons are randomly disabled
```

Example:

```text
100 neurons
↓
70 remain active
```

Because neurons cannot rely on specific neighboring neurons, the network learns more distributed representations.

Benefits:

* Reduces overfitting
* Improves generalization
* Encourages robust feature learning

Dropout is automatically disabled during evaluation mode.

---

# 📊 Validation Accuracy

Validation accuracy measures how well the model performs on unseen validation examples.

Higher validation accuracy generally indicates better generalization.

Monitoring validation accuracy helps determine whether additional training is beneficial.

---

# 📈 Training Loss Curve

The training loss visualization shows whether the model is learning.

A healthy curve usually looks like:

```text
High Loss
     ↓
Lower Loss
```

Loss should generally decrease over time.

---

# 📊 Confusion Matrix

A confusion matrix shows which classes are confused by the model.

Example:

```text
Actual Coat
Predicted Pullover
```

This helps identify weaknesses in classification performance.

Fashion-MNIST often shows confusion between:

* Shirts
* Pullovers
* Coats

because these categories share similar visual characteristics.

---

# 🖼️ Sample Predictions

The project generates prediction examples showing:

```text
Input Image
Predicted Class
```

These visualizations help verify that the model is making sensible predictions.

---

# 📈 Expected Results

Typical performance for this architecture:

```text
Validation Accuracy: 87% - 90%
Test Accuracy: 86% - 89%
```

Performance may vary slightly due to random initialization and train-validation splits.

---

# Key Takeaways

By completing this project, you will understand:

* How neural networks learn
* How forward propagation works
* How backpropagation updates weights
* Why loss functions matter
* How Adam optimization works
* Why validation sets are important
* What overfitting looks like
* How Dropout improves generalization
* How to evaluate classification models
* How Deep Learning models are trained in practice

This project serves as an excellent introduction to modern Deep Learning workflows and prepares you for more advanced architectures such as Convolutional Neural Networks (CNNs), Transfer Learning, and Computer Vision systems.

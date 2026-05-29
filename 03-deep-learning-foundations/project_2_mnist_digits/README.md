# 🟢 MNIST Digit Classifier

A beginner-friendly deep learning project that trains a neural network to recognize handwritten digits using PyTorch and the MNIST dataset.

This project teaches the core foundations of modern AI and deep learning through hands-on implementation and visualization.

---

# 📌 Project Overview

The goal of this project is to build an AI model capable of classifying handwritten digits from 0–9.

The model learns by analyzing thousands of handwritten examples and identifying patterns in the pixel data.

Example:

| Input Image     | Prediction |
| --------------- | ---------- |
| handwritten "3" | 3          |
| handwritten "7" | 7          |
| handwritten "9" | 9          |

---

# 🧠 AI Concepts Covered

This project introduces several foundational deep learning concepts:

* neural networks
* image classification
* tensors
* feedforward networks
* activation functions
* loss functions
* backpropagation
* optimization
* gradient descent
* model evaluation
* confusion matrices
* visualization

---

# 📁 Project Structure

```text
mnist_digit_classifier/
│
├── data/
│
├── outputs/
│   ├── digit_classifier.pth
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── predictions.png
│
├── utils/
│   └── model.py
│
├── train.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_2_mnist_digits

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

The script will:

* download the MNIST dataset
* train the neural network
* save the trained model
* generate visualizations
* evaluate model accuracy

If a trained model already exists:

```text
outputs/digit_classifier.pth
```

the program skips training and directly loads the saved model.

---

# 🧠 Understanding the MNIST Dataset

MNIST is one of the most famous datasets in machine learning.

It contains:

* 70,000 handwritten digit images
* grayscale images
* digits from 0–9

Each image is:

```text
28 × 28 pixels
```

That means each image contains:

```text
784 total pixels
```

Every pixel has a brightness value:

* 0 → black
* 255 → white

The neural network learns patterns from these pixel values.

---

# 🧠 What Is Image Classification?

Image classification means:

> Teaching a computer to recognize what an image contains.

In this project:

```text
Input Image → Neural Network → Predicted Digit
```

The network learns by seeing thousands of examples.

Over time, it becomes better at identifying patterns.

---

# 🧠 What Is a Neural Network?

A neural network is a system inspired by the human brain.

It consists of layers of connected neurons.

This project uses a:

```text
Feedforward Neural Network
```

Architecture:

```text
Input Layer (784)
        ↓
Hidden Layer (128)
        ↓
ReLU Activation
        ↓
Hidden Layer (64)
        ↓
ReLU Activation
        ↓
Output Layer (10)
```

---

# 🧠 Input Layer

The input layer receives the image data.

Since MNIST images are:

```text
28 × 28
```

they are flattened into:

```text
784 numbers
```

The network processes these numbers as features.

---

# 🧠 Hidden Layers

Hidden layers help the network learn patterns.

Different neurons may learn:

* edges
* curves
* loops
* diagonal lines
* digit shapes

As data moves deeper into the network:

* simple patterns combine into complex patterns
* the model learns better representations

---

# 🧠 Activation Functions

Neural networks require activation functions to learn complex relationships.

This project uses:

```python
nn.ReLU()
```

ReLU stands for:

```text
Rectified Linear Unit
```

Formula:

```math
f(x)=max(0,x)
```

Meaning:

* negative values become 0
* positive values stay unchanged

Without activation functions, neural networks would behave like simple linear equations and could not learn complicated patterns.

---

# 🧠 Output Layer

The final layer contains:

```text
10 neurons
```

One neuron for each digit:

```text
0–9
```

The model outputs scores for each digit.

The highest score becomes the prediction.

Example:

```text
[0.01, 0.02, 0.90, 0.01, ...]
```

Prediction:

```text
2
```

because digit 2 has the highest score.

---

# 🧠 What Is a Loss Function?

The loss function measures:

> How wrong the model's predictions are.

This project uses:

```python
nn.CrossEntropyLoss()
```

Cross entropy is commonly used for classification tasks.

If predictions are poor:

```text
high loss
```

If predictions improve:

```text
low loss
```

The goal of training is to minimize loss.

---

# 🧠 What Is Gradient Descent?

Gradient descent is the optimization process used to improve the model.

The model:

1. makes predictions
2. calculates error
3. adjusts weights slightly
4. repeats thousands of times

The network gradually improves after every batch.

---

# 🧠 What Is Backpropagation?

Backpropagation is how the neural network learns.

After calculating the loss:

```python
loss.backward()
```

PyTorch computes gradients for every weight in the network.

Gradients tell the model:

```text
How should each weight change to reduce error?
```

The optimizer then updates the weights.

---

# 🧠 What Is an Optimizer?

The optimizer updates the neural network weights.

This project uses:

```python
optim.Adam()
```

Adam is one of the most popular optimizers because it:

* converges quickly
* works well in practice
* automatically adapts learning rates

Think of the optimizer as:

```text
The learning engine of the neural network
```

---

# 🧠 What Is an Epoch?

An epoch means:

> One complete pass through the entire training dataset.

Example:

* Epoch 1 → random predictions
* Epoch 3 → learning shapes
* Epoch 5 → strong accuracy

The more epochs:

* the more the model learns
* but too many can cause overfitting

---

# 🧠 What Is Batch Training?

Instead of processing all images at once, the model processes small groups called batches.

Example:

```python
batch_size = 32
```

This means:

```text
32 images are processed together
```

Batching helps:

* reduce memory usage
* speed up training
* improve GPU efficiency

---

# 🧠 CPU vs GPU vs MPS

The project automatically detects available hardware.

### CPU

Standard processor.

Slower for deep learning.

---

### CUDA

NVIDIA GPU acceleration.

Much faster training.

---

### MPS

Apple Silicon GPU acceleration for Macs.

Uses Apple Metal Performance Shaders.

Useful for:

* M1
* M2
* M3 Macs

---

# 📈 Model Evaluation

After training, the model is tested on unseen images.

Accuracy is calculated using:

```text
correct predictions / total predictions
```

Typical results:

```text
96%–98% accuracy
```

---

# 📊 Confusion Matrix

The confusion matrix shows:

```text
Which digits the model confuses
```

Example:

* some 5s may look like 3s
* some 9s may look like 4s

The diagonal represents correct predictions.

---

# 🎨 Visualizations

The project generates several visual outputs.

---

## Training Loss

```text
outputs/training_loss.png
```

Shows how the loss decreases over time.

Lower loss indicates learning.

---

## Confusion Matrix

```text
outputs/confusion_matrix.png
```

Shows classification performance for every digit.

---

## Predictions

```text
outputs/predictions.png
```

Displays:

* input digit
* predicted digit
* true label

This helps visually inspect model performance.

---

# 🧠 Why This Project Matters

This project teaches the foundations behind:

* computer vision
* facial recognition
* OCR systems
* medical imaging AI
* autonomous vehicles
* large AI systems

The same core concepts used here scale into modern deep learning systems.

---

# 🚀 Possible Improvements

Future upgrades could include:

* Convolutional Neural Networks (CNNs)
* dropout regularization
* deeper architectures
* learning rate scheduling
* data augmentation
* real handwritten input support
* deployment with Flask/FastAPI

---

# 📚 Technologies Used

* Python
* PyTorch
* Torchvision
* Matplotlib
* Scikit-learn

---

# 🏁 Final Thoughts

This project is one of the best beginner introductions to deep learning.

By completing it, you learn:

* how neural networks learn
* how image classification works
* how training loops operate
* how optimization improves AI models
* how modern deep learning systems are built

This is the foundation for more advanced AI topics such as:

* CNNs
* Transformers
* GANs
* LLMs
* Computer Vision Systems
* Reinforcement Learning

---

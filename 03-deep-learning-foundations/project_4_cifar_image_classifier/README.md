# 🟡 Project 4 — CIFAR-10 Image Classifier (CNN Introduction)

This project is an introduction to **Convolutional Neural Networks (CNNs)** using PyTorch.

The goal is to build a model that can classify images from the **CIFAR-10 dataset**, which contains 10 different object categories like airplanes, cars, birds, cats, and trucks.

---

# 📂 Project Structure

```text
project_4_cifar_image_classifier/

│
├── data/
│
├── outputs/
│   ├── cifar_model.pth
│   ├── training_loss.png
│   ├── validation_accuracy.png
│   ├── confusion_matrix.png
│   └── predictions.png
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
cd 03-deep-learning-foundations/project_4_cifar_image_classifier

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

# 🎯 Goal of the Project

The purpose of this project is to understand how neural networks can learn visual patterns from data.

By the end, you will understand:
- How images are processed by neural networks
- How convolutional layers extract features
- How training works step-by-step
- How a model makes predictions on new images

---

# 🧠 What is CIFAR-10?

CIFAR-10 is a dataset containing:
- 60,000 images total
- 10 classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

Each image is:
- 32 × 32 pixels
- RGB (3 color channels)

---

# 🧠 Core AI Concepts Explained

## 1. What is a Neural Network?

A neural network is a system that learns patterns from data by adjusting internal parameters called **weights**.

Think of it like:
Input image → pattern recognition → prediction

---

## 2. Why CNNs (Convolutional Neural Networks)?

Regular neural networks struggle with images because:
- They ignore spatial structure
- They treat pixels independently

CNNs solve this by learning:
- edges
- shapes
- textures
- object parts

---

# 🧱 CNN Architecture in This Project

CNN = Feature Extractor + Classifier

---

# 🔍 Feature Extractor (Convolutional Layers)

The convolutional part of the model:

- Extracts visual patterns from images
- Builds hierarchical understanding (simple → complex features)

Layers used:
- Conv2d
- BatchNorm2d
- ReLU
- MaxPool2d

---

## 🧩 What is a Convolution?

A convolution applies a small filter across the image to detect patterns.

- Early layers detect edges
- Middle layers detect shapes
- Deep layers detect object parts

Think of it as scanning the image with a small sliding window.

---

## 📉 What is MaxPooling?

MaxPooling reduces the size of feature maps.

Example:
32×32 → 16×16 → 8×8 → 4×4

Benefits:
- reduces computation
- keeps strongest features
- improves robustness

---

## 🧪 What is Batch Normalization?

Batch normalization:
- stabilizes training
- speeds up convergence
- keeps values in a healthy range

It prevents unstable gradients and helps deep networks train better.

---

## ⚡ What is ReLU?

ReLU introduces non-linearity:

f(x) = max(0, x)

Why it matters:
- allows learning complex patterns
- avoids purely linear transformations

---

# 🧠 Classifier (Fully Connected Layers)

After convolution, the image is flattened into a vector.

This part:
- interprets extracted features
- maps them to class labels
- outputs predictions for 10 classes

Structure:
- Linear layer
- ReLU
- Dropout
- Linear output layer

---

## 🎯 Why Dropout?

Dropout randomly disables neurons during training.

Benefits:
- prevents overfitting
- forces generalization
- improves robustness

---

# 🔁 How Training Works

Training happens in epochs.

Each epoch:
1. Model sees images
2. Makes predictions
3. Computes loss (error)
4. Backpropagates gradients
5. Updates weights

---

## 📉 Loss Function

CrossEntropyLoss is used:

- measures prediction error
- compares predicted vs true labels
- lower is better

---

## ⚙️ Optimizer (Adam)

Adam optimizer:
- adapts learning rates automatically
- converges faster than standard gradient descent
- widely used in deep learning

---

# 📊 Validation

After each epoch:
- model is tested on unseen data
- accuracy is computed

This ensures:
- the model is learning patterns, not memorizing data

---

# 🧪 Testing the Model

After training:
- model is evaluated on test dataset
- final accuracy is computed

This gives the real-world performance.

---

# 📉 Confusion Matrix

A confusion matrix shows:
- correct predictions
- incorrect predictions
- which classes are confused

Example:
- cat vs dog confusion is common

---

# 🖼️ Sample Predictions

The model also visualizes predictions on test images:
- shows image
- shows predicted label

This helps interpret model behavior visually.

---

# 📦 Full Pipeline

Image → Convolution Layers → Feature Maps → Flatten → Fully Connected Layers → Prediction

---

# 🚀 Key Takeaways

You should now understand:
- how CNNs extract features
- how training works
- how models learn from images
- how predictions are made
- how performance is evaluated

---

# 🧠 Next Improvements

You can improve this project by:
- adding data augmentation
- using deeper networks (ResNet)
- tuning hyperparameters
- adding learning rate scheduling
- using transfer learning

---

# 🏁 Final Result

A CNN trained on CIFAR-10 that learns hierarchical image features and classifies objects into 10 categories.
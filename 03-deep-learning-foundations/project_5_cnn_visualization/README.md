# CNN Visualization Toolkit

A beginner-friendly Computer Vision project focused on understanding what Convolutional Neural Networks (CNNs) learn internally.

Instead of treating a CNN as a black box, this project visualizes the intermediate representations learned during training, including convolution filters, feature maps, and activation patterns.

The goal is to develop intuition for how neural networks transform raw pixels into meaningful visual concepts.

---

# Project Objectives

This project explores one of the most important questions in Deep Learning:

> What is a Convolutional Neural Network actually learning?

By training a CNN on the CIFAR-10 dataset and visualizing intermediate layers, we can observe how information flows through the network and how increasingly complex visual features emerge.

The project demonstrates:

* Convolutional Layers
* Feature Extraction
* Pooling Operations
* Activation Functions
* Learned Filters
* Feature Maps
* CNN Internal Representations

---

# Dataset

This project uses the CIFAR-10 dataset.

CIFAR-10 contains:

* 60,000 images
* 10 object categories
* 32×32 RGB images

Classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

The dataset is intentionally small, making it ideal for learning CNN fundamentals.

---

# Project Structure

```text
cnn-visualization-toolkit/

│
├── data/
│   └── CIFAR10/
│
├── outputs/
│   │
│   ├── feature_maps/
│   │   ├── layer1/
│   │   ├── layer2/
│   │   └── layer3/
│   │
│   ├── filters/
│   │   ├── conv1_filters.png
│   │   └── conv2_filters.png
│   │
│   ├── predictions/
│   │   └── sample_predictions.png
│   │
│   └── activation_maps/
│       └── activation_grid.png
│
├── utils/
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── feature_visualizer.py
│   └── filter_visualizer.py
│
├── train.py
├── visualize.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_5_cnn_visualization

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

# How To Run

## Step 1 — Train the CNN

```bash
python train.py
```

This will:

* Download CIFAR-10
* Train the CNN
* Save model weights

Output:

```text
cnn_model.pth
```

---

## Step 2 — Generate Visualizations

```bash
python visualize.py
```

This will generate:

```text
outputs/
├── filters/
├── feature_maps/
├── activation_maps/
└── predictions/
```

---

# Understanding Convolutional Neural Networks

Before examining visualizations, it is important to understand how CNNs work.

Traditional neural networks flatten images into one long vector:

```text
32 × 32 × 3
↓
3072 numbers
```

This destroys spatial information.

CNNs solve this problem by preserving image structure and learning local visual patterns.

---

# The CNN Architecture

The network consists of two major components:

```text
Image
 ↓
Feature Extractor
 ↓
Classifier
 ↓
Prediction
```

---

# Feature Extractor

The feature extractor is responsible for discovering visual patterns.

In our model:

```python
Conv2d
ReLU
MaxPool2d

Conv2d
ReLU
MaxPool2d

Conv2d
ReLU
MaxPool2d
```

Each convolution layer learns increasingly sophisticated visual features.

---

# Convolution Layers

The most important operation in a CNN is convolution.

A convolution layer contains learnable filters.

Example:

```text
3 × 3 Filter

[ 1  0 -1 ]
[ 1  0 -1 ]
[ 1  0 -1 ]
```

This filter acts as a vertical edge detector.

The filter slides across the image:

```text
Image
↓
Filter
↓
Dot Product
↓
Feature Map
```

Initially filters are random.

During training they automatically evolve into detectors for:

* edges
* textures
* shapes
* object parts

---

# What Filters Learn

Early filters often resemble:

```text
vertical edges
horizontal edges
color gradients
corners
```

These are the building blocks of vision.

Later layers combine these simple patterns into more meaningful structures.

---

# Feature Maps

A feature map is the output of a convolution.

Example:

```text
Input Image
↓
Edge Detector
↓
Feature Map
```

Bright regions indicate:

```text
strong activation
```

Dark regions indicate:

```text
weak activation
```

Feature maps reveal what information a particular filter finds important.

---

# ReLU Activation

After convolution:

```python
nn.ReLU()
```

is applied.

ReLU stands for:

```text
Rectified Linear Unit
```

Formula:

```text
max(0, x)
```

Behavior:

```text
Positive Values → Kept

Negative Values → Zero
```

Example:

```text
[-2, 4, -1, 8]

↓

[0, 4, 0, 8]
```

Benefits:

* introduces nonlinearity
* speeds up training
* prevents saturation issues

Without ReLU, deep networks would be unable to learn complex patterns.

---

# Pooling Layers

After activation:

```python
nn.MaxPool2d(2)
```

reduces spatial dimensions.

Example:

```text
4×4 Feature Map

[1 2 5 3]
[4 7 2 1]
[6 3 8 2]
[2 5 1 4]
```

After Max Pooling:

```text
[7 5]
[6 8]
```

Benefits:

* reduces computation
* reduces memory usage
* improves translation invariance
* focuses on strongest signals

---

# Hierarchical Feature Learning

CNNs learn visual concepts hierarchically.

---

## Layer 1

Detects simple patterns.

Examples:

```text
edges
corners
color transitions
textures
```

---

## Layer 2

Combines edges into structures.

Examples:

```text
fur patterns
wheels
windows
wings
```

---

## Layer 3

Combines structures into object parts.

Examples:

```text
animal faces
car bodies
ship structures
airplane shapes
```

---

## Final Layers

The classifier uses learned features to determine:

```text
Which class is most likely?
```

Example:

```text
Feature: Wings
Feature: Engine
Feature: Tail

↓

airplane
```

---

# Classifier

After feature extraction:

```text
128 × 4 × 4
```

is flattened into:

```text
2048 values
```

These values are passed through fully connected layers.

```python
Linear
ReLU
Linear
```

The classifier learns relationships between extracted features and object classes.

---

# Loss Function

The project uses:

```python
CrossEntropyLoss()
```

Cross entropy measures prediction error.

Good prediction:

```text
Low Loss
```

Bad prediction:

```text
High Loss
```

The objective of training is to minimize this loss.

---

# Optimizer

The model uses:

```python
Adam
```

Adam combines ideas from:

* Momentum
* Adaptive Learning Rates

Benefits:

* fast convergence
* stable training
* minimal tuning

For beginner projects, Adam is often the best default optimizer.

---

# What Is Being Visualized?

This project generates several visualization types.

---

## Learned Filters

Location:

```text
outputs/filters/
```

Shows:

```text
What each convolution kernel learned.
```

Examples:

* edge detectors
* color detectors
* texture detectors

---

## Feature Maps

Location:

```text
outputs/feature_maps/
```

Shows:

```text
How an image activates each filter.
```

You can observe how information changes from layer to layer.

---

## Activation Maps

Location:

```text
outputs/activation_maps/
```

Shows:

```text
Which image regions produce the strongest neural responses.
```

---

## Prediction Visualizations

Location:

```text
outputs/predictions/
```

Displays:

```text
Image
True Label
Predicted Label
```

Useful for understanding model behavior.

---

# Why This Project Matters

Many introductory Deep Learning projects focus exclusively on accuracy.

This project focuses on understanding.

By visualizing CNN internals, you learn:

* how convolution works
* what filters learn
* how feature hierarchies emerge
* how neural networks interpret images
* why CNNs outperform traditional neural networks on vision tasks

These concepts form the foundation for understanding advanced architectures such as:

* ResNet
* EfficientNet
* Vision Transformers (ViTs)
* YOLO
* Faster R-CNN
* Segment Anything (SAM)

---

# Future Improvements

Possible extensions:

### Grad-CAM

Visual explanation heatmaps showing where the network looked before making a prediction.

### Misclassification Analysis

Visualize incorrectly classified images.

### Deeper Architectures

Replace the custom CNN with:

* ResNet18
* ResNet34
* EfficientNet

### Interactive Dashboard

Build a Streamlit application for exploring activations in real time.

### Feature Embeddings

Use PCA or t-SNE to visualize learned feature representations.

---

# Key Takeaway

A CNN does not directly learn objects.

Instead, it learns a hierarchy of increasingly complex visual features:

```text
Pixels
 ↓
Edges
 ↓
Textures
 ↓
Patterns
 ↓
Object Parts
 ↓
Objects
```

This hierarchical feature learning is the core idea behind modern Computer Vision systems.

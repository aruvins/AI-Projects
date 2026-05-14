# Vision Transformers (ViTs)

A deep learning computer vision project focused on implementing and benchmarking Vision Transformers (ViTs) against traditional Convolutional Neural Networks (CNNs) for image classification tasks.

This project explores:

* patch embeddings
* self-attention mechanisms
* transformer encoders
* convolutional neural networks
* transfer learning
* model benchmarking

using PyTorch and Hugging Face Transformers.

---

# Project Overview

Computer vision models have traditionally relied on Convolutional Neural Networks (CNNs) to process images. CNNs dominated computer vision for over a decade because of their ability to efficiently learn spatial patterns such as:

* edges
* textures
* object structures

More recently, Vision Transformers (ViTs) introduced a transformer-based approach to computer vision inspired by transformer architectures used in natural language processing.

This project compares:

* CNN architectures
* Transformer-based vision models

to better understand:

* inductive biases
* self-attention
* global context modeling
* scaling behavior
* computational tradeoffs

---

# Project Goals

## Vision Transformers

* Implement transformer-based image classification
* Explore patch embeddings
* Understand self-attention in vision
* Fine-tune pretrained ViTs

---

## Convolutional Neural Networks

* Train CNN baselines
* Understand convolution operations
* Explore feature extraction
* Compare CNN performance against transformers

---

## Benchmarking

Evaluate:

* inference speed
* training performance
* parameter efficiency
* classification accuracy

between CNNs and ViTs.

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* Hugging Face Transformers
* NumPy
* Matplotlib
* tqdm
* scikit-learn

---

# Project Structure

```bash id="ylv2p4"
Project_3_Vision_Transformers/
│
├── data/
│
├── models/
│   ├── cnn_model.py
│   └── vit_model.py
│
├── train/
│   ├── train_resnet.py
│   ├── train_vit.py
│   └── compare_models.py
│
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── visualize_attention.py
│
├── checkpoints/
│
├── requirements.txt
└── README.md
```

---

# Dataset

This project uses the CIFAR-10 dataset.

Dataset:

[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com)

CIFAR-10 contains:

* 60,000 images
* 10 classes
* 32×32 RGB images

Classes include:

* airplanes
* cars
* birds
* cats
* dogs
* ships
* trucks

The dataset is automatically downloaded using Torchvision.

---

# Understanding Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks designed for image processing.

They use:

* convolution operations
* feature hierarchies
* local receptive fields

to learn visual patterns.

---

# How CNNs Work

CNNs process images through several stages.

---

# 1. Convolution Layers

A convolution layer applies learnable filters to an image.

Example:

```text id="ommtx7"
3×3 filter slides across image
```

The filter detects patterns such as:

* edges
* corners
* textures

As layers deepen, CNNs learn:

* object parts
* semantic structures
* high-level features

---

# 2. Activation Functions

Nonlinear activation functions such as:

```python id="4z1wzr"
ReLU()
```

allow networks to learn complex patterns.

---

# 3. Pooling Layers

Pooling reduces spatial dimensions.

Example:

```text id="svhgrv"
224×224 → 112×112
```

Benefits:

* reduces computation
* improves invariance
* extracts dominant features

---

# 4. Fully Connected Layers

Final layers map learned features to classification outputs.

Example:

```text id="od0mhh"
Image → "cat"
```

---

# CNN Strengths

CNNs excel because they:

* efficiently learn local patterns
* require fewer parameters
* perform well on smaller datasets
* exploit spatial locality

However:

* CNN receptive fields are local
* long-range relationships are harder to model

---

# Understanding Vision Transformers (ViTs)

Vision Transformers apply transformer architectures directly to images.

Instead of:

* convolution filters
* local kernels

ViTs process images as sequences of patches using:

* self-attention
* token embeddings
* transformer encoders

Paper:

[Vision Transformer Paper](https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com)

---

# How Vision Transformers Work

---

# 1. Patch Extraction

Input image:

```text id="hrp0od"
224 × 224 × 3
```

is divided into smaller patches.

Example:

```text id="g1jll5"
16 × 16 patches
```

This creates:

```text id="jlwm6j"
14 × 14 = 196 patches
```

Each patch becomes a token.

---

# 2. Patch Embeddings

Each image patch is flattened into a vector.

Example:

```text id="jlwm6m"
16 × 16 × 3 → embedding vector
```

This is analogous to:

* word embeddings in NLP transformers

---

# 3. Positional Encoding

Transformers do not inherently understand spatial order.

Positional embeddings encode:

* patch positions
* spatial relationships

allowing the model to preserve image structure.

---

# 4. Self-Attention

Self-attention allows each patch to attend to all other patches.

This enables:

* global context understanding
* long-range spatial reasoning
* holistic image modeling

Unlike CNNs:

* attention is not constrained to local receptive fields

---

# 5. Transformer Encoder

Patch sequences pass through transformer blocks consisting of:

* multi-head attention
* feed-forward networks
* layer normalization

These layers iteratively refine image representations.

---

# Vision Transformer Pipeline

```text id="jlwm6u"
Input Image
      │
      ▼
Patch Extraction
      │
      ▼
Patch Embeddings
      │
      ▼
Positional Encoding
      │
      ▼
Transformer Encoder
      │
      ▼
Classification Head
      │
      ▼
Predicted Class
```

---

# CNN vs Vision Transformer

| CNN                         | Vision Transformer          |
| --------------------------- | --------------------------- |
| Local receptive fields      | Global attention            |
| Convolution kernels         | Self-attention              |
| Strong inductive bias       | Data-driven learning        |
| Efficient on small datasets | Better scaling behavior     |
| Faster training             | Higher compute requirements |

---

# How This Project Works

This project implements:

* ResNet18 CNN baseline
* pretrained Vision Transformer

for image classification.

---

# CNN Model

The CNN baseline uses:

```python id="jlwm71"
torchvision.models.resnet18
```

ResNet introduces:

* residual connections
* deep feature learning
* stable optimization

---

# Vision Transformer Model

The ViT model uses:

```python id="jlwm74"
ViTForImageClassification
```

from Hugging Face Transformers.

The pretrained model:

* extracts image embeddings
* applies transformer attention
* predicts image classes

---

# Training Pipeline

The training workflow includes:

1. Load CIFAR-10 dataset
2. Resize images
3. Convert images to tensors
4. Forward pass through model
5. Compute classification loss
6. Backpropagation
7. Update weights
8. Evaluate performance

---

# Transfer Learning

Both models use pretrained weights.

Benefits:

* faster convergence
* higher accuracy
* reduced training time

Pretraining allows the models to leverage:

* large-scale visual representations
* learned image semantics

---

# Benchmarking

The project benchmarks:

* inference speed
* training efficiency
* classification performance

between:

* CNNs
* Vision Transformers

This helps illustrate:

* architectural tradeoffs
* scaling behavior
* transformer efficiency

---

# Running the Project

---

# 1. Create Virtual Environment

```bash id="jlwm7d"
python3.11 -m venv venv
source venv/bin/activate
```

---

# 2. Install Dependencies

## requirements.txt

```txt id="jlwm7f"
torch
torchvision
transformers
datasets
matplotlib
numpy
tqdm
scikit-learn
```

Install:

```bash id="jlwm7h"
pip install -r requirements.txt
```

---

# 3. Train Vision Transformer

```bash id="jlwm7k"
python train/train_vit.py
```

---

# 4. Train CNN Baseline

```bash id="jlwm7m"
python train/train_resnet.py
```

---

# 5. Compare Models

```bash id="jlwm7o"
python train/compare_models.py
```

---

# Expected Learning Outcomes

This project explores:

* Convolutional Neural Networks
* Vision Transformers
* Self-attention
* Patch embeddings
* Transfer learning
* Transformer encoders
* Model benchmarking
* Image classification pipelines

---

# Future Improvements

## Advanced Vision Models

Explore:

* DeiT
* Swin Transformer
* ConvNeXt
* EfficientNet
* Hybrid CNN-Transformer models

---

# Attention Visualization

Visualize:

* patch attention maps
* transformer focus regions
* spatial attention behavior

---

# Larger Datasets

Experiment with:

* CIFAR-100
* ImageNet
* Food101
* Oxford Pets

---

# References

## Vision Transformer Paper

[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com)

---

## Hugging Face Transformers

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index?utm_source=chatgpt.com)

---

## PyTorch

[PyTorch](https://pytorch.org/?utm_source=chatgpt.com)

---

## DeiT

[DeiT Repository](https://github.com/facebookresearch/deit?utm_source=chatgpt.com)

---

## Swin Transformer

[Swin Transformer](https://github.com/microsoft/Swin-Transformer?utm_source=chatgpt.com)

---

# Resume Project Description

Implemented Vision Transformer (ViT) architectures for image classification and benchmarked transformer-based vision models against traditional CNN architectures. Explored patch embeddings, self-attention mechanisms, positional encoding, and transformer encoders using PyTorch and Hugging Face Transformers. Fine-tuned pretrained ViT models and evaluated classification accuracy, inference performance, and architectural tradeoffs relative to ResNet baselines.

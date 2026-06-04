# 🔵 Project 9 — Transfer Learning Image Classifier

## 🎯 Goal

Build a high-performance image classifier using a pretrained deep neural network.

Instead of training a convolutional neural network from scratch, we leverage knowledge learned from millions of images and transfer that knowledge to a new task.

This project introduces one of the most important ideas in modern AI:

> Most production machine learning systems use transfer learning rather than training models from scratch.

---

# What You Will Learn

This project teaches:

- Transfer Learning
- Pretrained Models
- ResNet Architecture
- Feature Extraction
- Fine-Tuning
- Freezing Layers
- Convolutional Neural Networks
- Image Classification
- Confusion Matrices
- Model Evaluation

By the end of this project you will understand how modern computer vision systems can achieve high accuracy with relatively little training.

---

# Project Structure

```text
project_9_transfer_learning_classifier/

│
├── main.py
│
├── models/
│   └── transfer_model.py
│
├── utils/
│   ├── dataset.py
│   ├── trainer.py
│   ├── predictor.py
│   └── visualize.py
│
├── outputs/
│   ├── training_loss.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── predictions.png
│   └── transfer_model.pth
│
├── requirements.txt
│
└── README.md
```

---


# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_9_transfer_learning_classifier

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

# 4. Run the Project

```bash
python main.py
```

---

# Dataset

This project uses the CIFAR-10 dataset.

CIFAR-10 contains:

- 60,000 color images
- 10 classes
- 32 × 32 resolution

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

Example images:

```text
Airplane      Cat          Truck
   ✈️           🐱            🚚
```

Although CIFAR-10 is relatively small, it is a classic benchmark for image classification.

---

# Why Training From Scratch Is Expensive

Suppose we wanted to train a CNN ourselves.

The network must learn:

```text
Edges
 ↓
Corners
 ↓
Textures
 ↓
Shapes
 ↓
Object Parts
 ↓
Objects
```

This requires:

- large datasets
- long training times
- significant compute

Modern image models are often trained on millions of images.

For example:

```text
ImageNet

≈ 14 million images
≈ 22,000 categories
```

Training from scratch can take days or weeks.

---

# Transfer Learning

Transfer learning solves this problem.

Instead of training a model from random initialization:

```text
Random Weights
      ↓
Train Everything
      ↓
New Model
```

we start with:

```text
Pretrained Model
      ↓
Reuse Knowledge
      ↓
Adapt To New Task
```

The model already understands visual concepts such as:

```text
Edges
Textures
Patterns
Shapes
Object Parts
```

We simply teach it our new classes.

---

# Intuition

Imagine teaching someone to identify dogs.

You do not teach:

```text
What is an edge?
What is a curve?
What is a shape?
```

They already know these concepts.

Instead you teach:

```text
This combination of features = dog
```

Transfer learning works the same way.

---

# What Is ResNet?

This project uses ResNet18.

ResNet stands for:

```text
Residual Network
```

Introduced in the landmark paper:

"Deep Residual Learning for Image Recognition"

by Kaiming He et al. (2015).

---

# The Problem With Deep Networks

As neural networks become deeper:

```text
10 Layers
50 Layers
100 Layers
```

training becomes difficult.

Gradients can vanish.

Performance may stop improving.

Sometimes deeper models perform worse.

---

# Residual Connections

ResNet introduced shortcut connections.

Instead of learning:

```text
Output
```

a layer learns:

```text
Difference From Input
```

Diagram:

```text
Input --------+
              |
              v
          Conv Layer
              |
              +
              |
           Output
```

This allows gradients to flow more easily through very deep networks.

---

# Why ResNet Became Important

ResNet made it practical to train:

```text
18 Layers
34 Layers
50 Layers
101 Layers
152 Layers
```

deep neural networks.

Many modern vision systems are built upon ideas introduced by ResNet.

---

# Pretrained Models

TorchVision provides models already trained on ImageNet.

Example:

```python
models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
```

These weights have already learned rich visual representations.

---

# Transfer Learning Workflow

Our pipeline:

```text
Input Image
      ↓
Pretrained ResNet18
      ↓
Feature Extraction
      ↓
Classifier Layer
      ↓
Prediction
```

---

# Feature Extraction

The pretrained network acts as a feature extractor.

Example:

```text
Image
 ↓
ResNet
 ↓
Feature Vector
```

Instead of raw pixels, the classifier receives meaningful features.

These features may encode:

```text
Shape
Texture
Edges
Object Structure
```

---

# Freezing Layers

Initially we freeze the pretrained network.

```python
for param in model.parameters():
    param.requires_grad = False
```

This means:

```text
ResNet Weights
      ↓
DO NOT UPDATE
```

Only the final classifier learns.

Benefits:

- faster training
- less memory usage
- reduced overfitting

---

# Classifier Head

The original ResNet predicts ImageNet classes.

We replace the final layer.

Original:

```text
1000 Classes
```

New:

```text
10 CIFAR Classes
```

Example:

```python
model.fc = nn.Linear(
    num_features,
    10
)
```

This new layer learns our task.

---

# Fine-Tuning

After training the classifier head, we can optionally unfreeze some layers.

```python
for param in model.parameters():
    param.requires_grad = True
```

Now the entire network adapts.

This process is called:

```text
Fine-Tuning
```

---

# Feature Extraction vs Fine-Tuning

Feature Extraction:

```text
Pretrained Features
        +
Train New Classifier
```

Fast and stable.

---

Fine-Tuning:

```text
Pretrained Features
        +
Adjust Entire Network
```

Potentially higher accuracy.

Requires more computation.

---

# Training Process

For each batch:

```text
Images
   ↓
Forward Pass
   ↓
Predictions
   ↓
Loss
   ↓
Backpropagation
   ↓
Weight Updates
```

This process repeats for every epoch.

---

# Cross Entropy Loss

Classification uses:

```python
nn.CrossEntropyLoss()
```

Cross entropy measures:

```text
How wrong are the predictions?
```

Lower values indicate better performance.

---

# Adam Optimizer

Weights are updated using:

```python
optim.Adam(...)
```

Adam combines:

```text
Momentum
+
Adaptive Learning Rates
```

making training faster and more stable.

---

# Model Evaluation

Accuracy is computed using:

```text
Correct Predictions
-------------------
Total Predictions
```

Example:

```text
900 Correct
1000 Total

Accuracy = 90%
```

---

# Confusion Matrix

Accuracy alone can hide problems.

A confusion matrix shows:

```text
Actual Class
      vs
Predicted Class
```

Example:

```text
            Predicted

          Cat   Dog

Actual Cat 90    10
Actual Dog 15    85
```

This reveals which classes are being confused.

---

# Prediction Visualization

The project visualizes model predictions.

Each image displays:

```text
Predicted Label
True Label
```

This helps verify model behavior.

---

# Generated Outputs

After training, the outputs directory contains:

```text
outputs/

training_loss.png
accuracy_curve.png
confusion_matrix.png
predictions.png
transfer_model.pth
```

---

## training_loss.png

Shows learning progress.

```text
Loss
 ^
 |
 |\
 | \
 |  \
 |   \
 +----------> Epoch
```

Loss should generally decrease.

---

## accuracy_curve.png

Tracks model accuracy.

```text
Accuracy
 ^
 |
 |     /
 |   /
 | /
 +----------> Epoch
```

Accuracy should generally increase.

---

## confusion_matrix.png

Displays class-level performance.

Useful for identifying weaknesses.

---

## predictions.png

Shows sample predictions.

Helps visually inspect model quality.

---

## transfer_model.pth

Saved model weights.

Allows future inference without retraining.

---

# Expected Results

Feature Extraction Only:

```text
85% – 90%
```

Fine-Tuning:

```text
90% – 95%
```

depending on hyperparameters and training time.

---

# Key Takeaways

This project introduces one of the most impactful ideas in modern machine learning.

You learned:

- Why pretrained models work
- How transfer learning reduces training cost
- How ResNet learns visual features
- Why freezing layers is useful
- How fine-tuning improves performance
- How modern image classifiers are built

Most state-of-the-art computer vision systems today use transfer learning in some form.

Understanding this workflow is an important step toward:

- Object Detection
- Semantic Segmentation
- Vision Transformers (ViTs)
- Multimodal Models
- Large-Scale Foundation Models

and modern AI engineering.

---

## References

### Transfer Learning

1. **A Survey on Transfer Learning**
   Sinno Jialin Pan, Qiang Yang (2010)
   One of the foundational survey papers introducing transfer learning concepts.

   Paper: [https://ieeexplore.ieee.org/document/5288526](https://ieeexplore.ieee.org/document/5288526)

2. **Transfer Learning for Computer Vision Tutorial (PyTorch)**
   Official PyTorch tutorial demonstrating feature extraction and fine-tuning.

   [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

### ResNet

3. **Deep Residual Learning for Image Recognition**
   Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)

   Introduced Residual Networks (ResNet), one of the most influential architectures in deep learning.

   Paper: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

4. **Original ResNet Implementation (TorchVision)**

   [https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

---

### Convolutional Neural Networks

5. **Gradient-Based Learning Applied to Document Recognition**
   Yann LeCun et al. (1998)

   Introduced LeNet-5, one of the earliest successful convolutional neural networks.

   Paper: [http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

---

### ImageNet

6. **ImageNet Classification with Deep Convolutional Neural Networks**
   Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (2012)

   The AlexNet paper that popularized deep learning for computer vision.

   Paper: [https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

7. **ImageNet Dataset**

   [https://www.image-net.org/](https://www.image-net.org/)

---

### CIFAR-10 Dataset

8. **CIFAR-10 Dataset**

   [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

9. **Learning Multiple Layers of Features from Tiny Images**
   Alex Krizhevsky (2009)

   Technical report describing CIFAR-10 and CIFAR-100.

   [https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

---

### Optimization

10. **Adam: A Method for Stochastic Optimization**
    Diederik Kingma, Jimmy Ba (2014)

    Introduced the Adam optimizer used in this project.

    Paper: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

---

### PyTorch Documentation

11. **PyTorch Documentation**

    [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

12. **TorchVision Documentation**

    [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)

13. **TorchVision Pretrained Models**

    [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

---

### Recommended Reading

14. **The Illustrated CNN**

    [https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)

15. **CS231n: Convolutional Neural Networks for Visual Recognition**
    Stanford University

    [https://cs231n.stanford.edu/](https://cs231n.stanford.edu/)

16. **Dive Into Deep Learning**

    [https://d2l.ai/](https://d2l.ai/)

17. **Deep Learning**
    Ian Goodfellow, Yoshua Bengio, Aaron Courville

    Online version:

    [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

---

### Suggested Next Papers

After completing this project, the natural progression is:

* ResNet (2015)
* DenseNet (2017)
* EfficientNet (2019)
* Vision Transformer (2020)
* CLIP (2021)

#### DenseNet

[https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)

#### EfficientNet

[https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)

#### Vision Transformer (ViT)

[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

#### CLIP

[https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

These papers trace the evolution from classical CNN-based transfer learning toward modern foundation models and multimodal AI systems.

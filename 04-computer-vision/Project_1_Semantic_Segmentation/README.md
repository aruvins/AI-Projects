# Semantic Segmentation with U-Net

A deep learning computer vision project focused on semantic segmentation using a custom U-Net architecture implemented in PyTorch. This project explores encoder-decoder convolutional neural networks for pixel-wise image segmentation using the Oxford-IIIT Pet Dataset.

---

# Project Overview

Semantic segmentation is a computer vision task where every pixel in an image is classified into a category. Unlike image classification, which predicts a single label for an entire image, segmentation predicts dense pixel-level outputs.

In this project, a U-Net model is trained to identify and segment pets from background regions in images.

The project includes:

* A custom U-Net implementation in PyTorch
* Data augmentation pipelines using Albumentations
* Oxford-IIIT Pet Dataset integration through Torchvision
* Dice Score and IoU evaluation metrics
* Training and inference pipelines
* Visualization of segmentation predictions

---

# Example Task

Input Image:

```text
Image of a dog or cat
```

Model Output:

```text
Binary segmentation mask
1 = pet
0 = background
```

The network learns to predict which pixels belong to the foreground object.

---

# Key Concepts

## Semantic Segmentation

Semantic segmentation assigns a class label to every pixel in an image.

Applications include:

* Medical imaging
* Autonomous vehicles
* Satellite imaging
* Robotics
* Scene understanding

---

## U-Net Architecture

U-Net is an encoder-decoder convolutional neural network originally developed for biomedical image segmentation.

Paper:

[U-Net Paper](https://arxiv.org/abs/1505.04597?utm_source=chatgpt.com)

The architecture consists of:

### 1. Encoder (Contracting Path)

The encoder extracts hierarchical features from the image.

Operations:

* Convolution
* Batch normalization
* ReLU activation
* Max pooling

As depth increases:

* Spatial dimensions decrease
* Semantic understanding increases

Example:

```text
256x256 → 128x128 → 64x64 → 32x32
```

---

### 2. Bottleneck

The bottleneck contains the deepest learned representation of the image.

It captures:

* High-level semantics
* Object structure
* Contextual information

---

### 3. Decoder (Expanding Path)

The decoder reconstructs spatial resolution using:

* Transposed convolutions
* Upsampling
* Feature concatenation

The decoder converts compressed features back into a dense segmentation mask.

---

### 4. Skip Connections

One of the most important ideas in U-Net.

Encoder features are concatenated with decoder features:

```python
torch.cat([decoder_feature, encoder_feature], dim=1)
```

Benefits:

* Preserves fine-grained spatial information
* Improves localization accuracy
* Prevents information loss during downsampling

Without skip connections:

* object boundaries become blurry
* segmentation quality degrades significantly

---

# Project Architecture

```text
Input Image
      │
      ▼
Encoder (Downsampling)
      │
      ▼
Bottleneck
      │
      ▼
Decoder (Upsampling)
      │
      ▼
Segmentation Mask
```

---

# Dataset

This project uses the Oxford-IIIT Pet Dataset from Torchvision.

Dataset:

[Torchvision Oxford-IIIT Pet Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html?utm_source=chatgpt.com)

The dataset contains:

* Cats and dogs
* Pixel-level segmentation masks
* Foreground/background labels

Torchvision automatically downloads and organizes the dataset.

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* Albumentations
* NumPy
* Matplotlib
* OpenCV

---

# Project Structure

```bash
semantic-segmentation-unet/
│
├── models/
│   └── unet.py
│
├── utils/
│   ├── metrics.py
│   └── transforms.py
│
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

---

# How the Project Works

## 1. Dataset Loading

The Oxford-IIIT Pet dataset is loaded using Torchvision:

```python
from torchvision.datasets import OxfordIIITPet
```

Each sample contains:

* RGB image
* Segmentation mask

The trimap masks are converted into binary masks:

```python
mask = (mask == 1).astype(np.float32)
```

Where:

* 1 = foreground object
* 0 = background

---

## 2. Data Augmentation

Albumentations is used to improve generalization.

Augmentations include:

* Horizontal flipping
* Rotation
* Brightness/contrast adjustment
* Resizing
* Normalization

This helps the model learn robust spatial features.

---

## 3. Model Forward Pass

The image passes through:

### Encoder

Extracts semantic features.

### Bottleneck

Captures compressed representation.

### Decoder

Restores spatial resolution.

### Final Convolution

```python
nn.Conv2d(64, 1, kernel_size=1)
```

Produces:

* 1-channel segmentation mask

---

## 4. Loss Function

The project uses:

```python
nn.BCEWithLogitsLoss()
```

This combines:

* sigmoid activation
* binary cross entropy

for stable binary segmentation training.

---

## 5. Evaluation Metrics

### Dice Score

Measures overlap quality between prediction and ground truth.

Formula:

```text
Dice = 2TP / (2TP + FP + FN)
```

Widely used in medical imaging.

---

### IoU (Intersection over Union)

Measures segmentation overlap accuracy.

Formula:

```text
IoU = Intersection / Union
```

Higher IoU indicates better segmentation quality.

---

# Training Pipeline

Training workflow:

1. Load dataset
2. Apply augmentations
3. Forward pass through U-Net
4. Compute loss
5. Backpropagation
6. Update weights
7. Evaluate Dice and IoU

The model is trained over multiple epochs until segmentation quality improves.

---

# Inference Pipeline

Inference steps:

1. Load trained model
2. Load image
3. Resize and normalize
4. Generate segmentation mask
5. Apply sigmoid threshold
6. Visualize prediction

The final prediction shows the segmented foreground object.

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

---

## Run Inference

```bash
python inference.py
```

---

# Results

The model learns:

* object boundaries
* foreground/background separation
* spatial localization

Typical metrics on Oxford Pet:

* Dice Score: 0.85+
* IoU Score: 0.75+

depending on training duration and augmentation strategy.

---

# Challenges Encountered

## Channel Mismatch in Decoder

During implementation, skip connection concatenation doubled feature channels:

```text
64 + 64 = 128 channels
```

This required the final decoder convolution to reduce:

```python
DoubleConv(128, 64)
```

This debugging process helped reinforce understanding of U-Net skip connections and feature map dimensions.

---

# Future Improvements

Potential extensions include:

## Advanced Architectures

* Attention U-Net
* UNet++
* DeepLabV3+
* SegFormer

---

## Training Improvements

* Dice Loss
* Mixed precision training
* Learning rate schedulers
* Early stopping

---

## Dataset Extensions

* Medical imaging segmentation
* Satellite segmentation
* Autonomous driving datasets
* Multi-class segmentation

---

# Learning Outcomes

Through this project, the following concepts were explored:

* Semantic segmentation
* Encoder-decoder architectures
* Convolutional neural networks
* Skip connections
* Data augmentation
* Pixel-wise prediction
* Segmentation metrics
* Deep learning training pipelines

---

# References

## U-Net Paper

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597?utm_source=chatgpt.com)

## PyTorch

[PyTorch](https://pytorch.org/?utm_source=chatgpt.com)

## Torchvision Datasets

[Torchvision Datasets](https://pytorch.org/vision/stable/datasets.html?utm_source=chatgpt.com)

## Albumentations

[Albumentations](https://albumentations.ai/?utm_source=chatgpt.com)

## segmentation-models-pytorch

[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch?utm_source=chatgpt.com)

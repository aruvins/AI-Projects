# Segment Anything Model (SAM)

A computer vision project focused on interactive and zero-shot image segmentation using Meta AI’s Segment Anything Model (SAM). This project explores prompt-based segmentation systems powered by foundation vision models and Vision Transformers.

---

# Project Overview

The Segment Anything Model (SAM) is a foundation segmentation model developed by Meta Platforms AI Research.

Unlike traditional segmentation models that require task-specific training, SAM performs segmentation using prompts such as:

* mouse clicks
* bounding boxes
* masks
* points

This project demonstrates:

* interactive segmentation
* automatic mask generation
* real-time segmentation systems
* zero-shot object segmentation

using pretrained SAM checkpoints.

---

# What Makes SAM Important

Traditional segmentation models:

* require labeled datasets
* need retraining for new tasks
* specialize in narrow domains

SAM changes this paradigm by introducing:

* promptable segmentation
* foundation vision models
* zero-shot segmentation

SAM can segment objects it has never explicitly trained on.

This makes SAM similar to how large language models generalized NLP.

---

# Core Goals

## Interactive Segmentation

Segment objects using:

* positive clicks
* negative clicks
* bounding boxes

---

## Zero-Shot Segmentation

Generate segmentation masks without training on a custom dataset.

---

## Automatic Mask Generation

Detect and segment all objects in an image automatically.

---

## Real-Time Segmentation

Apply SAM to:

* webcam feeds
* live video
* interactive applications

---

# Project Structure

```bash id="0dtrng"
Project_2_Segment_Anything_Model/
│
├── checkpoints/
│   └── sam_vit_b_01ec64.pth
│
├── images/
│   └── sample.jpg
│
├── sam/
│   ├── interactive_segmentation.py
│   ├── automatic_masks.py
│   ├── box_segmentation.py
│   └── webcam_segmentation.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Segment Anything Model (SAM)

---

# How SAM Works

SAM is composed of three major neural network components.

---

# 1. Image Encoder

The image encoder converts an input image into a dense feature embedding.

SAM uses:

* Vision Transformers (ViT)

Variants include:

* ViT-B
* ViT-L
* ViT-H

The encoder extracts:

* spatial features
* semantic representations
* object boundaries

The image is transformed into a high-dimensional latent representation.

---

# 2. Prompt Encoder

The prompt encoder processes user inputs such as:

* points
* clicks
* boxes
* masks

Example prompt:

```python id="1m1jsi"
input_point = np.array([[500, 375]])
```

The prompt encoder converts these inputs into embeddings that guide segmentation.

This allows SAM to behave interactively.

---

# 3. Mask Decoder

The mask decoder combines:

* image embeddings
* prompt embeddings

to generate segmentation masks.

Outputs include:

* candidate masks
* confidence scores
* segmentation logits

The decoder predicts pixel-wise object boundaries.

---

# SAM Architecture Pipeline

```text id="tn6t6v"
Input Image
      │
      ▼
Vision Transformer Encoder
      │
      ▼
Image Embeddings
      │
      ▼
Prompt Encoder
      │
      ▼
Mask Decoder
      │
      ▼
Segmentation Masks
```

---

# Project Setup

install 

---

# How This Project Works

The project integrates pretrained SAM checkpoints into multiple segmentation workflows.

---

# 1. Interactive Segmentation

Users click on objects to generate masks.

Example:

```python id="e0iylz"
input_point = np.array([[500, 375]])
input_label = np.array([1])
```

Where:

* 1 = foreground object
* 0 = background point

SAM predicts:

* object boundaries
* segmentation masks
* multiple candidate masks

This simulates real annotation systems.

---

# 2. Automatic Mask Generation

The project uses:

```python id="n4hljlwm"
SamAutomaticMaskGenerator
```

to automatically segment every object in an image.

This enables:

* automatic labeling
* object discovery
* dataset generation

without human interaction.

---

# 3. Bounding Box Segmentation

Users provide a bounding box:

```python id="jlwmvn"
input_box = np.array([
    x1, y1, x2, y2
])
```

SAM segments the object inside the box.

Applications include:

* annotation tools
* object extraction
* image editing systems

---

# 4. Real-Time Webcam Segmentation

The webcam pipeline:

* captures live frames
* applies prompt-based segmentation
* overlays predicted masks

This creates:

* interactive AI applications
* real-time segmentation systems
* robotics-style computer vision pipelines

---

# Prompt-Based Segmentation

One of the most important innovations in SAM is promptability.

Instead of training:

* prompts guide segmentation dynamically

This is analogous to:

* prompts in large language models

but applied to computer vision.

---

# Zero-Shot Learning

SAM performs zero-shot segmentation because:

* it generalizes to unseen objects
* it does not require retraining
* it learns universal segmentation priors

This represents a major shift in computer vision research.

---

# Interactive Segmentation Example

Input:

* image
* click prompt

Output:

* predicted object mask

Visualization pipeline:

```text id="aqjlwm"
Original Image
      +
Point Prompt
      +
Predicted Mask
      +
Overlay Visualization
```

---

# Automatic Mask Generation Example

SAM can segment:

* people
* animals
* objects
* background structures

simultaneously without prompts.

This is useful for:

* autonomous systems
* robotics
* scene understanding
* dataset annotation

---

# Model Checkpoints

This project uses Meta’s pretrained SAM checkpoints.

## ViT-B

Recommended for:

* local development
* laptops
* experimentation

Checkpoint download:

[SAM ViT-B Checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth?utm_source=chatgpt.com)

---

# Running the Project

## Install Dependencies

```bash id="1yjlwm"
pip install torch torchvision matplotlib opencv-python numpy
```

Install SAM:

```bash id="5n79kn"
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install SAM Checkpoint:

```bash 

curl -L -o sam_vit_b_01ec64.pth \                                        
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
---

# Interactive Segmentation

```bash id="6hwn64"
python sam/interactive_segmentation.py
```

---

# Automatic Mask Generation

```bash id="l85acg"
python sam/automatic_masks.py
```

---

# Bounding Box Segmentation

```bash id="jlwmmi"
python sam/box_segmentation.py
```

---

# Webcam Segmentation

```bash id="c9k6l4"
python sam/webcam_segmentation.py
```

---

# Apple Silicon Optimization

The project supports Apple Metal acceleration using:

```python id="epjlwm"
torch.backends.mps.is_available()
```

This enables GPU acceleration on:

* M1
* M2
* M3
* Apple Silicon Macs


---

# Important Concepts Learned

This project explores:

* Foundation vision models
* Vision Transformers
* Prompt engineering for vision
* Interactive AI systems
* Zero-shot learning
* Segmentation architectures
* Real-time computer vision
* Prompt encoding systems
* Mask decoding pipelines

---

# Real-World Applications

SAM-like systems are used in:

* medical imaging
* robotics
* autonomous vehicles
* annotation platforms
* augmented reality
* image editing
* video segmentation

---

# Grounded SAM

Combines:

* object detection
* segmentation
* text prompting

Example:

```text id="jlwmg0"
"segment the dog"
```

Project:

[Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything?utm_source=chatgpt.com)

---

# SAM2

Meta later released SAM2 for:

* video segmentation
* object tracking
* temporal memory

Repository:

[SAM2 GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com)

---

# References

## SAM Research Paper

[Segment Anything Paper](https://arxiv.org/abs/2304.02643?utm_source=chatgpt.com)

---

## SAM GitHub Repository

[Segment Anything GitHub](https://github.com/facebookresearch/segment-anything?utm_source=chatgpt.com)

---

## PyTorch

[PyTorch](https://pytorch.org/?utm_source=chatgpt.com)

---

## OpenCV

[OpenCV](https://opencv.org/?utm_source=chatgpt.com)

---

# Resume Project Description

Developed interactive computer vision systems using Meta AI’s Segment Anything Model (SAM) for prompt-driven and zero-shot image segmentation. Implemented interactive point prompting, automatic mask generation, bounding-box segmentation, and real-time webcam segmentation pipelines using Vision Transformers and PyTorch. Explored foundation vision models, prompt encoding systems, and real-time segmentation applications.

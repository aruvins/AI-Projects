# README.md

# 3D Computer Vision From Scratch

A complete educational implementation of stereo vision, Structure from Motion (SfM), and Neural Radiance Fields (NeRFs) built entirely from scratch.

This repository focuses on understanding the mathematics, geometry, optimization, and rendering techniques behind modern 3D computer vision systems instead of relying on pretrained models or high-level frameworks.

The goal of this project is to deeply understand how machines reconstruct and reason about 3D scenes from 2D images.

---

# Table of Contents

1. Introduction
2. Project Goals
3. Pipeline Overview
4. Stereo Vision
5. Structure From Motion (SfM)
6. Neural Radiance Fields (NeRF)
7. Mathematical Foundations
8. Repository Structure
9. Installation
10. Running the Project
11. Datasets
12. Future Improvements
13. Learning Outcomes
14. References

---

# Introduction

3D computer vision enables machines to understand the spatial structure of the world using images.

Unlike traditional 2D computer vision, 3D vision attempts to recover:
- depth
- geometry
- camera motion
- scene structure
- volumetric representations

This project implements three major 3D vision systems:

| Component | Purpose |
|---|---|
| Stereo Vision | Recover depth from two images |
| Structure From Motion | Recover camera motion + sparse geometry |
| NeRF | Learn continuous neural scene representations |

---

# Project Goals

The primary goal of this repository is educational depth.

Instead of using:
- pretrained depth estimation networks
- COLMAP
- pretrained NeRF pipelines
- OpenCV black-box geometry functions

this project implements the core mathematics and algorithms manually.

The emphasis is on:
- linear algebra
- projective geometry
- optimization
- multi-view geometry
- differentiable rendering

---

# Pipeline Overview

The project is divided into three phases:

```text
2D Images
    ↓
Stereo Matching
    ↓
Depth Estimation
    ↓
Point Cloud Generation
    ↓
Feature Matching
    ↓
Camera Pose Estimation
    ↓
Triangulation
    ↓
Sparse 3D Reconstruction
    ↓
Neural Scene Representation
    ↓
Volume Rendering
    ↓
Novel View Synthesis

---

# 1. Stereo Vision

Stereo vision estimates depth using two images captured from slightly different viewpoints.

This is inspired by human binocular vision.

---

# Core Idea

Objects appear at different horizontal positions in left and right camera images.

This horizontal displacement is called:

## Disparity

Closer objects produce larger disparity.

Farther objects produce smaller disparity.

The system estimates:

* how much each pixel shifts
* converts that shift into physical depth

---

# Stereo Geometry

## Camera Setup

Two cameras are separated by a fixed distance called the baseline.

```text
Left Camera  -------- baseline --------  Right Camera
```

A 3D point projects differently onto each image plane.

---

# Disparity Equation

[
d = x_l - x_r
]

Where:

* (x_l) = pixel position in left image
* (x_r) = pixel position in right image
* (d) = disparity

---

# Depth Equation

Depth is computed using triangulation:

[
Z = \frac{fB}{d}
]

Where:

* (Z) = depth
* (f) = focal length
* (B) = camera baseline
* (d) = disparity

---

# Stereo Matching Algorithm

The project implements block matching manually.

For every pixel:

1. Extract a small patch from the left image
2. Search horizontally in the right image
3. Compare similarity using SSD
4. Select the best match

---

# Sum of Squared Differences (SSD)

Similarity is computed using:

[
SSD = \sum (I_l - I_r)^2
]

Lower SSD means better correspondence.

---

# Why This Works

Stereo vision works because:

* the same object appears in both cameras
* nearby objects shift more between views
* geometry constrains matching to horizontal lines

This is known as:

## Epipolar Geometry

---

# Point Cloud Generation

After depth estimation, pixels are projected into 3D coordinates.

## Projection Equations

[
X = \frac{(u-c_x)Z}{f_x}
]

[
Y = \frac{(v-c_y)Z}{f_y}
]

[
Z = depth
]

Where:

* (u,v) = pixel coordinates
* (c_x,c_y) = optical center
* (f_x,f_y) = focal lengths

The resulting 3D coordinates form a point cloud.

---

# 2. Structure From Motion (SfM)

Structure From Motion reconstructs:

* camera movement
* sparse scene geometry

from multiple images.

SfM is one of the foundational algorithms behind:

* photogrammetry
* SLAM
* AR/VR
* robotics
* autonomous vehicles

---

# SfM Pipeline

The pipeline consists of:

```text
Image Features
    ↓
Feature Matching
    ↓
Essential Matrix Estimation
    ↓
Camera Pose Recovery
    ↓
Triangulation
    ↓
Sparse 3D Reconstruction
```

---

# Feature Detection

The project uses ORB features.

Features are:

* corners
* edges
* high-information regions

Good features are stable across viewpoints.

---

# Feature Matching

Descriptors from both images are compared using Hamming distance.

Correct matches represent the same physical 3D point.

---

# Essential Matrix

The essential matrix encodes relative camera geometry.

[
x_2^T E x_1 = 0
]

Where:

* (x_1) = point in image 1
* (x_2) = point in image 2
* (E) = essential matrix

---

# What the Essential Matrix Represents

The essential matrix captures:

* camera rotation
* camera translation

between two views.

It constrains where corresponding points may exist.

---

# Eight-Point Algorithm

The project estimates the essential matrix using the normalized eight-point algorithm.

Steps:

1. Normalize image points
2. Construct linear system
3. Solve using SVD
4. Enforce rank-2 constraint

---

# Singular Value Decomposition (SVD)

SVD decomposes a matrix into:

[
A = U \Sigma V^T
]

This is heavily used throughout:

* essential matrix estimation
* triangulation
* pose recovery

---

# Camera Pose Recovery

After estimating (E), the camera pose is recovered.

The decomposition yields:

* rotation matrix (R)
* translation vector (t)

These describe how the camera moved between frames.

---

# Triangulation

Triangulation reconstructs 3D coordinates from multiple 2D observations.

Each image ray defines a line in 3D space.

The intersection of these rays estimates the original 3D point.

---

# Projection Matrix

A camera projection matrix is defined as:

[
P = K[R|t]
]

Where:

* (K) = intrinsic matrix
* (R) = rotation
* (t) = translation

---

# Sparse Reconstruction

The final output is a sparse point cloud representing:

* scene geometry
* camera motion
* spatial structure

---

# 3. Neural Radiance Fields (NeRF)

NeRF is a neural rendering technique that represents scenes continuously using a neural network.

Instead of explicit geometry:

* meshes
* voxels
* point clouds

NeRF learns a function:

[
F(x,y,z,\theta,\phi) \rightarrow (r,g,b,\sigma)
]

Where:

* spatial coordinates map to:

  * RGB color
  * density

---

# Why NeRF Is Important

NeRF revolutionized:

* view synthesis
* neural rendering
* 3D scene representation

It enables:

* photorealistic novel views
* continuous scene representations
* differentiable rendering

---

# NeRF Pipeline

```text
Camera Rays
    ↓
Sample 3D Points
    ↓
Positional Encoding
    ↓
MLP Prediction
    ↓
Density + RGB
    ↓
Volume Rendering
    ↓
Rendered Image
```

---

# Ray Generation

Each pixel corresponds to a ray projected into 3D space.

The camera emits rays into the scene.

Points are sampled along each ray.

---

# Positional Encoding

NeRF uses sinusoidal positional encodings.

Without encoding:

* MLPs struggle with high-frequency detail

Encoding transforms coordinates into higher-dimensional features.

---

# Positional Encoding Equation

[
\gamma(x) =
[
\sin(2^0x), \cos(2^0x),
\sin(2^1x), \cos(2^1x),
...
]
]

This enables the network to model:

* sharp edges
* textures
* fine geometry

---

# Neural Network Architecture

The NeRF model is a fully-connected MLP.

Input:

* encoded 3D coordinates

Output:

* RGB color
* density

---

# Density

Density determines:

* how much matter exists at a point
* how much light is absorbed

Higher density means:

* opaque regions
* surfaces
* objects

---

# Volume Rendering

NeRF renders images using differentiable volume rendering.

The final pixel color is accumulated along the ray.

---

# Volume Rendering Equation

[
C(r)=\sum_i T_i(1-e^{-\sigma_i\delta_i})c_i
]

Where:

* (T_i) = transmittance
* (\sigma_i) = density
* (\delta_i) = distance between samples
* (c_i) = RGB color

---

# Transmittance

Transmittance models how much light survives traveling through space.

If density is high:

* less light passes through

If density is low:

* rays continue further

---

# Why NeRF Works

NeRF learns:

* geometry
* appearance
* lighting

implicitly through optimization.

Instead of storing explicit surfaces, the network stores a continuous volumetric function.

---

# Repository Structure

```text
3d-computer-vision/
│
├── data/
│
├── stereo_depth/
│   ├── disparity.py
│   ├── depth.py
│   ├── pointcloud.py
│   └── run_stereo.py
│
├── sfm/
│   └── reconstruction.py
│
├── nerf/
│   ├── encoding.py
│   ├── model.py
│   ├── rays.py
│   ├── render.py
│   └── train.py
│
├── common/
├── outputs/
└── README.md
```

---

# Installation

```bash
git clone <repo-url>

cd computer-Vision/Project_7_3D_Computer_Vision

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

# Requirements

```txt
numpy
opencv-python
matplotlib
open3d
torch
torchvision
tqdm
imageio
scipy
```
---
# Download Necessary Dataset
```bash
python dataset_downloader.py
```

---

# Running Stereo Vision

```bash
cd stereo_depth
python run_stereo.py
```

Outputs:

* disparity map
* depth map
* point cloud

---

# Running Structure From Motion

```bash
cd sfm
python reconstruction.py
```

Outputs:

* recovered camera poses
* sparse 3D reconstruction

---

# Running NeRF

```bash
cd nerf
python train.py
```

Outputs:

* trained NeRF model
* rendered RGB outputs

---

# Recommended Datasets

## Stereo Vision

* KITTI
* Middlebury
* ETH3D

## Structure From Motion

* Fountain Dataset
* Temple Dataset
* Custom phone images

## NeRF

* TinyNeRF
* Blender Synthetic Dataset

---

# Future Improvements

## Stereo Vision

* Semi-global matching
* Census transform
* Subpixel estimation
* GPU acceleration

## SfM

* RANSAC
* Bundle adjustment
* Dense reconstruction
* Multi-view optimization

## NeRF

* Hierarchical sampling
* View-direction conditioning
* Hash encodings
* Gaussian splatting
* Instant-NGP acceleration

---

# Learning Outcomes

This project teaches:

* projective geometry
* camera systems
* epipolar geometry
* multi-view reconstruction
* linear algebra
* optimization
* neural rendering
* differentiable graphics

---

# References

## Stereo Vision

* [Hartley & Zisserman — Multiple View Geometry](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)
* [Szeliski — Computer Vision Algorithms and Applications](https://szeliski.org/Book/)

## Structure From Motion

* [Multiple View Geometry in Computer Vision](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)
* [Visual SLAM tutorials](https://learnopencv.com/tag/visual-slam/)

## NeRF

* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934)
* [TinyNeRF implementation](https://github.com/volunt4s/TinyNeRF-pytorch)

---

# Suggested Resume Bullet

Implemented a complete 3D computer vision pipeline from scratch including stereo depth estimation, Structure from Motion, sparse 3D reconstruction, and Neural Radiance Fields (NeRFs) using multi-view geometry, triangulation, and differentiable volume rendering.

```
```

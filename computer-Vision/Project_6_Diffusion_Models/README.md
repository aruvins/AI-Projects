# Diffusion Models & Generative Vision

A complete generative AI and computer vision project built with Python, PyTorch, and Hugging Face Diffusers.

This project explores modern diffusion models and generative vision systems including:

- Text-to-Image Generation
- Stable Diffusion Fine-Tuning
- Image Editing / Inpainting
- Image-to-Image Translation
- Synthetic Dataset Generation
- Interactive Gradio UI

---

# Features

## Text-to-Image Generation
Generate images directly from text prompts using Stable Diffusion.

Example:
- “A futuristic cyberpunk city”
- “A dragon flying over snowy mountains”

---

## Stable Diffusion Fine-Tuning
Train or fine-tune Stable Diffusion on custom datasets using paired images and captions.

Applications:
- Custom art styles
- Product generation
- Character consistency
- Domain-specific image generation

---

## Image Editing / Inpainting
Edit specific parts of an image using masks and prompts.

Example:
- Replace the sky with a galaxy
- Add objects into a scene
- Remove unwanted objects

---

## Image-to-Image Generation
Transform images into new styles.

Examples:
- Oil painting
- Watercolor
- Anime
- Cyberpunk

---

## Synthetic Dataset Generation
Automatically generate labeled datasets for machine learning.

Applications:
- Object detection
- Robotics
- Autonomous driving
- Medical imaging

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- Gradio
- PIL
- OpenCV
- NumPy

---

# Project Structure

```bash
Project_6_Diffusion_Models/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── train/
│   ├── generated/
│   ├── input.jpg
│   └── mask.png
│
├── models/
│   └── checkpoints/
│
├── outputs/
│   ├── images/
│   └── edits/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── utils.py
│   ├── text_to_image.py
│   ├── train_sd.py
│   ├── image_editing.py
│   ├── image_to_image.py
│   ├── synthetic_dataset.py
│   ├── create_example.py
│   └── gradio_app.py
│
└── venv/
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <your_repo_url>
cd Project_6_Diffusion_Models
```

---

## 2. Create a Virtual Environment

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# requirements.txt

```txt
torch
torchvision
diffusers
transformers
accelerate
opencv-python
pillow
matplotlib
numpy
datasets
safetensors
xformers
gradio
```

---

# Apple Silicon (M1/M2/M3) Setup

If you are using Apple Silicon, update:

## src/config.py

```python
DEVICE = "mps"
```

Also replace:

```python
torch.float16
```

with:

```python
torch.float32
```

inside:
- text_to_image.py
- image_editing.py
- image_to_image.py
- synthetic_dataset.py
- gradio_app.py

---

# How Diffusion Models Work

Diffusion models learn by:

1. Adding noise to images
2. Learning how to remove the noise
3. Generating new images by reversing the process

---

# Stable Diffusion Architecture

Stable Diffusion consists of:

## 1. CLIP Text Encoder
Converts prompts into embeddings.

## 2. Variational Autoencoder (VAE)
Compresses images into latent space.

## 3. U-Net
Predicts and removes noise.

## 4. Scheduler
Controls the denoising process.

---

# Running the Project

---

# 1. Text-to-Image Generation

Generate images from prompts.

## Run

```bash
python src/text_to_image.py
```

## Example Prompt

```python
prompt = "A futuristic cyberpunk city"
```

## Output

```bash
outputs/images/generated.png
```

---

# 2. Image Editing / Inpainting

Edit parts of an image using masks.

## Run

```bash
python src/image_editing.py
```

## Input Files

```bash
data/input.jpg
data/mask.png
```

## Mask Rules

- White = edit
- Black = preserve

## Example Prompt

```python
"Replace the sky with a colorful galaxy"
```

## Output

```bash
outputs/edits/edited.png
```

---

# 3. Image-to-Image Generation

Transform existing images into new styles.

## Run

```bash
python src/image_to_image.py
```

## Example Prompt

```python
"Convert this image into an oil painting"
```

## Output

```bash
outputs/images/transformed.png
```

---

# 4. Stable Diffusion Fine-Tuning

Train Stable Diffusion on a custom dataset.

---

# Dataset Format

## data/train/

```bash
image1.jpg
image2.jpg
captions.txt
```

## captions.txt

```txt
image1.jpg|a futuristic robot
image2.jpg|a sports car
```

---

# Run Training

```bash
python src/train_sd.py
```

---

# Training Pipeline

The training script:

1. Loads images
2. Encodes images into latent space
3. Adds random noise
4. Trains the U-Net to predict the noise
5. Computes MSE loss
6. Updates weights
7. Saves checkpoints

---

# Checkpoints

Saved to:

```bash
models/checkpoints/
```

---

# 5. Synthetic Dataset Generation

Generate synthetic datasets automatically.

## Run

```bash
python src/synthetic_dataset.py
```

## Output

```bash
data/generated/
```

Contains:
- generated images
- annotations.csv

---

# 6. Launch Gradio Web App

Interactive UI for image generation.

## Run

```bash
python src/gradio_app.py
```

Open:

```txt
http://127.0.0.1:7860
```

---

# Example Prompts

## Sci-Fi
- A futuristic neon city
- A spaceship entering hyperspace

## Fantasy
- A dragon flying over mountains
- A medieval castle at sunset

## Realistic
- Cinematic portrait photography
- Professional fashion photoshoot

---

# Common Errors

---

# ModuleNotFoundError: No module named 'diffusers'

Fix:

```bash
pip install diffusers transformers accelerate
```

---

# CUDA / GPU Errors

If using MacBook:

```python
DEVICE = "mps"
```

If using NVIDIA GPU:

```python
DEVICE = "cuda"
```

---

# Slow Generation

Increase performance with:
- smaller image sizes
- fewer inference steps
- xformers optimization

---

# Future Improvements

## Intermediate
- Negative prompts
- Batch generation
- Better schedulers
- Metadata saving

## Advanced
- LoRA fine-tuning
- DreamBooth
- ControlNet
- Training from scratch
- Custom DDPM implementation
- Video diffusion
- 3D diffusion

---

# Learning Outcomes

This project teaches:

- Diffusion Models
- Generative AI
- Stable Diffusion
- U-Net Architectures
- Latent Space Modeling
- Image Synthesis
- Inpainting
- Prompt Conditioning
- Fine-Tuning
- Synthetic Data Generation

---

# References

## Documentation
- PyTorch
- Hugging Face Diffusers
- Transformers
- Gradio

## Papers
- Denoising Diffusion Probabilistic Models (DDPM)
- High-Resolution Image Synthesis with Latent Diffusion Models

---

# Resume Description

Built multiple generative computer vision systems using Stable Diffusion, including text-to-image generation, image editing pipelines, synthetic dataset generation, and custom fine-tuning workflows using PyTorch and Hugging Face Diffusers.
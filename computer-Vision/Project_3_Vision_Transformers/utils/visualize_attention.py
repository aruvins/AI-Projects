import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import ViTModel
from transformers import ViTImageProcessor


# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# Load pretrained ViT
model = ViTModel.from_pretrained(
    "google/vit-base-patch16-224",
    output_attentions=True
)

model.to(DEVICE)

model.eval()


# Image processor
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)


# Load image
image_path = "sample.jpg"

image = Image.open(image_path).convert("RGB")


# Process image
inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    k: v.to(DEVICE)
    for k, v in inputs.items()
}


# Forward pass
with torch.no_grad():

    outputs = model(**inputs)


# Attention tensors
attentions = outputs.attentions


# Last transformer layer
last_attention = attentions[-1]


# Shape:
# [batch, heads, tokens, tokens]
print(last_attention.shape)


# Average attention across heads
attention_map = last_attention.mean(dim=1)


# Remove CLS token
attention_map = attention_map[0, 1:, 1:]


# Average token attention
attention_map = attention_map.mean(dim=0)


# Convert to numpy
attention_map = attention_map.cpu().numpy()


# Reshape into patch grid
num_patches = int(np.sqrt(attention_map.shape[0]))

attention_map = attention_map.reshape(
    num_patches,
    num_patches
)


# Resize to image dimensions
attention_map = cv2.resize(
    attention_map,
    image.size
)


# Normalize
attention_map = (
    attention_map - attention_map.min()
) / (
    attention_map.max() - attention_map.min()
)


# Plot
plt.figure(figsize=(12, 6))


# Original image
plt.subplot(1, 2, 1)

plt.imshow(image)

plt.title("Original Image")

plt.axis("off")


# Attention heatmap
plt.subplot(1, 2, 2)

plt.imshow(image)

plt.imshow(
    attention_map,
    cmap="jet",
    alpha=0.5
)

plt.title("ViT Attention Map")

plt.axis("off")


plt.tight_layout()

plt.show()
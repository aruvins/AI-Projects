import matplotlib
matplotlib.use("TkAgg")

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)


# Device
# if torch.backends.mps.is_available():
#     DEVICE = "mps"
# elif torch.cuda.is_available():
#     DEVICE = "cuda"
# else:
    # DEVICE = "cpu"

DEVICE = "cpu"


# Load SAM
sam = sam_model_registry["vit_b"](
    checkpoint="checkpoints/sam_vit_b_01ec64.pth"
)

sam.to(device=DEVICE)


# Automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)


# Load image
image = cv2.imread("images/sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Generate masks
masks = mask_generator.generate(image)

print(f"Generated {len(masks)} masks")


# Plot image
plt.figure(figsize=(12, 12))

plt.imshow(image)


# Overlay masks
for mask_data in masks:

    mask = mask_data["segmentation"]

    plt.imshow(mask, alpha=0.35)


plt.title("Automatic Mask Generation")

plt.axis("off")

plt.savefig("automatic_masks_result.png")
import os
import random

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.unet import UNet


# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# Load model
model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        "unet_model.pth",
        map_location=DEVICE
    )
)

model.eval()


# Dataset directory
IMAGE_DIR = "./data/oxford-iiit-pet/images"

# Number of samples
NUM_IMAGES = 10


# Random sample images
image_files = random.sample(
    os.listdir(IMAGE_DIR),
    NUM_IMAGES
)


# Create large vertical figure
fig, axes = plt.subplots(
    NUM_IMAGES,
    3,
    figsize=(15, NUM_IMAGES * 4)
)

fig.suptitle(
    "U-Net Segmentation Results",
    fontsize=20
)


for idx, image_name in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original = image.copy()

    # Preprocess
    resized = cv2.resize(image, (256, 256))
    resized = resized.astype(np.float32) / 255.0

    tensor = torch.tensor(
        np.transpose(resized, (2, 0, 1))
    ).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():

        pred = model(tensor)

        pred = torch.sigmoid(pred)

        pred = (pred > 0.5).float()

    mask = pred.squeeze().cpu().numpy()

    # Create overlay
    overlay = original.copy()

    overlay = cv2.resize(
        overlay,
        (256, 256)
    )

    overlay[mask > 0.5] = [255, 0, 0]

    # Plot original
    axes[idx, 0].imshow(original)
    axes[idx, 0].set_title("Original")
    axes[idx, 0].axis("off")

    # Plot mask
    axes[idx, 1].imshow(mask, cmap="gray")
    axes[idx, 1].set_title("Predicted Mask")
    axes[idx, 1].axis("off")

    # Plot overlay
    axes[idx, 2].imshow(overlay)
    axes[idx, 2].set_title("Overlay")
    axes[idx, 2].axis("off")


plt.tight_layout()

# Enables scrolling in many environments
plt.show()
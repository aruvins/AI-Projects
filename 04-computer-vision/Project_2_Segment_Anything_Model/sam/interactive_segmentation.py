import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry
from segment_anything import SamPredictor


# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# Load SAM
sam = sam_model_registry["vit_b"](
    checkpoint="checkpoints/sam_vit_b_01ec64.pth"
)

sam.to(device=DEVICE)

predictor = SamPredictor(sam)


# Load image
image_path = "images/sample.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)


# Point prompt
input_point = np.array([[500, 375]])
input_label = np.array([1])


# Predict masks
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)


# Best mask
best_mask = masks[np.argmax(scores)]


# Visualization
plt.figure(figsize=(10, 10))

plt.imshow(image)

plt.imshow(best_mask, alpha=0.5)

plt.scatter(
    input_point[:, 0],
    input_point[:, 1],
    color="red",
    s=100
)

plt.title("Interactive Segmentation")

plt.axis("off")

plt.show()
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
image = cv2.imread("images/sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)


# Bounding box
input_box = np.array([
    200,
    100,
    700,
    600
])


# Predict
masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=True
)


best_mask = masks[np.argmax(scores)]


# Visualization
plt.figure(figsize=(10, 10))

plt.imshow(image)

plt.imshow(best_mask, alpha=0.5)

# Draw box
x1, y1, x2, y2 = input_box

plt.gca().add_patch(
    plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        edgecolor="red",
        facecolor="none",
        linewidth=2
    )
)

plt.title("Bounding Box Segmentation")

plt.axis("off")

plt.show()
plt.savefig("./results/box_segmentations.png")
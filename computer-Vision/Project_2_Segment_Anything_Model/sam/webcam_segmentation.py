import cv2
import torch
import numpy as np

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


# Webcam
cap = cv2.VideoCapture(0)

print("Press Q to quit")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    predictor.set_image(rgb)

    h, w, _ = rgb.shape

    # Center point
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = masks[0]

    overlay = frame.copy()

    overlay[mask] = [0, 255, 0]

    cv2.circle(
        overlay,
        (w // 2, h // 2),
        5,
        (0, 0, 255),
        -1
    )

    cv2.imshow(
        "SAM Webcam Segmentation",
        overlay
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()
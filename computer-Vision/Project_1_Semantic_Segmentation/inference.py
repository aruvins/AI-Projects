import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = UNet().to(DEVICE)
model.load_state_dict(torch.load("unet_model.pth", map_location=DEVICE))
model.eval()


image_path = "sample.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original = image.copy()

image = cv2.resize(image, (256, 256))
image = image.astype(np.float32) / 255.0

image = np.transpose(image, (2, 0, 1))
image = torch.tensor(image).unsqueeze(0).to(DEVICE)


with torch.no_grad():
    pred = model(image)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

mask = pred.squeeze().cpu().numpy()


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Predicted Mask")

plt.show()
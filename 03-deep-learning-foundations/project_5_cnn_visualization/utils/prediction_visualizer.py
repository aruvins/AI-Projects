import os
import torch
import numpy as np
import matplotlib.pyplot as plt

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


def visualize_predictions(model, images, labels, device):
    os.makedirs("outputs/predictions", exist_ok=True)

    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))

        _, preds = torch.max(outputs, 1)

    images = images.cpu().numpy()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10, 6)
    )

    for i, ax in enumerate(axes.flat):
        img = np.transpose(
            images[i],
            (1, 2, 0)
        )

        img = img * 0.5 + 0.5
        ax.imshow(img)

        true_label = CLASSES[labels[i]]
        pred_label = CLASSES[preds[i].item()]

        ax.set_title(f"T: {true_label}\nP: {pred_label}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/predictions/sample_predictions.png")
    plt.show()
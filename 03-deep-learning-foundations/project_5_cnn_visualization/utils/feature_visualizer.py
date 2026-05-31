import torch
import matplotlib.pyplot as plt
import os

def visualize_feature_maps(model, image, device):
    os.makedirs("./outputs/feature_maps", exist_ok=True)
    activations = []
    hooks = []
    
    def hook(module, input, output):
        activations.append(output.detach())
    
    
    for layer in model.features:
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))
    image = image.unsqueeze(0).to(device)
    model(image)

    for h in hooks:
        h.remove()

    for idx, activation in enumerate(activations):
        fmap = activation[0]
        fig, axes = plt.subplots(
            4,
            4,
            figsize=(8, 8)
        )

        for i, ax in enumerate(axes.flat):
            ax.imshow(
                fmap[i].cpu(),
                cmap = "viridis"
            )
            ax.axis("off")

        plt.tight_layout
        plt.savefig(
            f"./outputs/feature_maps/layer_{idx+1}.png"
        )
        plt.show()
import os
import torch
import matplotlib.pyplot as plt


def visualize_activation_map(model,image,device):
    os.makedirs("outputs/activation_maps",exist_ok=True)

    activations = []

    def hook(module,input,output):
        activations.append(output.detach())

    last_conv = None
    for layer in model.features:
        if isinstance(layer,torch.nn.Conv2d):
            last_conv = layer

    handle = last_conv.register_forward_hook(hook)
    image_batch = image.unsqueeze(0).to(device)
    model(image_batch)
    handle.remove()

    fmap = activations[0][0]
    activation_map = fmap.mean(dim=0)

    activation_map = (activation_map.cpu().numpy())

    plt.figure(figsize=(6, 6))
    plt.imshow(activation_map,cmap="hot")
    plt.colorbar()
    plt.title("Average Activation Map")
    plt.axis("off")
    plt.savefig("outputs/activation_maps/activation_map.png")
    plt.show()
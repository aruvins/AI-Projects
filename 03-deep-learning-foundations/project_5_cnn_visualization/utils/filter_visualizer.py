import matplotlib.pyplot as plt
import os

def visualize_filters(conv_layer, filename):
    os.makedirs("./outputs/filters", exist_ok=True)
    filters = conv_layer.weight.data.cpu()
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i >= filters.shape[0]:
            break
        f = filters[i]
        f = (f - f.min()) / (
            f.max() - f.min() + 1e-8
        )

        if f.shape[0] == 3:
            ax.imshow(f.permute(1, 2, 0).numpy())
        else:
            ax.imshow(f[0].numpy(),cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"./outputs/filters/{filename}")
    plt.show()
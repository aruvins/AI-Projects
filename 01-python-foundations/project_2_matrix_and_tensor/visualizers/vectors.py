import matplotlib.pyplot as plt
import numpy as np
from core.io_utils import save_plot

def plot_vectors(vectors, save_path=None, show=True):
    plt.figure()

    origin = [0, 0]

    for i, v in enumerate(vectors):
        plt.quiver(
            *origin, v[0], v[1],
            angles='xy',
            scale_units='xy',
            scale=1,
            label=f"v{i}",
            color="blue"
        )

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.legend()
    plt.title("Vector Visualization")

    if save_path:
        save_plot(save_path)

    if show:
        plt.show()
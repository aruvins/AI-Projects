import numpy as np
import matplotlib.pyplot as plt
from core.io_utils import save_plot

def apply_transform(matrix, vectors):
    return [np.dot(matrix, v) for v in vectors]


def plot_transform(original, transformed, save_path=None, show=True):
    plt.figure()

    for v in original:
        plt.quiver(0, 0, v[0], v[1], color="blue")

    for v in transformed:
        plt.quiver(0, 0, v[0], v[1], color="red")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.title("Blue = Original | Red = Transformed")

    if save_path:
        save_plot(save_path)

    if show:
        plt.show()
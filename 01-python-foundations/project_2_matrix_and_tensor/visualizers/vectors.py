import matplotlib.pyplot as plt
import numpy as np

def plot_vector(v, color="blue", label="vector"):
    origin = [0, 0]

    plt.quiver(*origin, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color, label=label)

def show_vectors(vectors):
    plt.figure()

    for i, v in enumerate(vectors):
        plot_vector(v, label=f"v{i}")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.legend()
    plt.title("Vector Visualization")
    plt.show()
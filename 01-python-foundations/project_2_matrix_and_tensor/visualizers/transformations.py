import numpy as np
import matplotlib.pyplot as plt

def apply_transform(matrix, vectors):
    transformed = []
    for v in vectors:
        transformed.append(np.dot(matrix, v))
    return transformed

def plot_transform(original, transformed):
    plt.figure()

    for v in original:
        plt.quiver(0, 0, v[0], v[1], color="blue")

    for v in transformed:
        plt.quiver(0, 0, v[0], v[1], color="red")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.title("Blue = Original | Red = Transformed")
    plt.show()
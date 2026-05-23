import numpy as np
import matplotlib.pyplot as plt
from core.io_utils import save_plot

def show_eigen(matrix, save_path=None, show=True):
    eigvals, eigvecs = np.linalg.eig(matrix)

    print("\nEigenvalues:")
    print(eigvals)

    print("\nEigenvectors:")
    print(eigvecs)

    plt.figure()

    origin = np.array([0, 0])

    for i in range(len(eigvals)):
        vec = np.real(eigvecs[:, i])

        vec = vec / np.linalg.norm(vec)

        plt.quiver(
            *origin,
            vec[0],
            vec[1],
            angles='xy',
            scale_units='xy',
            scale=1,
            color="blue",
            label=f"eigenvector {i}"
        )

        scaled = vec * np.real(eigvals[i])

        plt.quiver(
            *origin,
            scaled[0],
            scaled[1],
            angles='xy',
            scale_units='xy',
            scale=1,
            color="red",
            alpha=0.6,
            label=f"scaled {i}"
        )

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.legend()
    plt.title("Eigenvectors (blue) and Scaled Effect (red)")

    if save_path:
        save_plot(save_path)

    if show:
        plt.show()
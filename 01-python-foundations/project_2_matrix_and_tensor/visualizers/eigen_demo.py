import numpy as np
import matplotlib.pyplot as plt

def show_eigen(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)

    print("Eigenvalues:", eigvals)
    print("Eigenvectors:\n", eigvecs)

    plt.figure()

    for i in range(len(eigvals)):
        v = eigvecs[:, i] * eigvals[i]
        plt.quiver(0, 0, v[0], v[1], scale_units='xy', scale=1)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.title("Eigenvector Scaling Effect")
    plt.show()
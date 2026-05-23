import numpy as np
from visualizers.eigen_demo import show_eigen

def run():
    matrix = np.array([
        [1.5, 0.5],
        [0.5, 1.5]
    ])

    show_eigen(
        matrix,
        save_path="outputs/eigen.png",
        show=True
    )

if __name__ == "__main__":
    run()
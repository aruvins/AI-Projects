import numpy as np
from visualizers.vectors import plot_vectors

def run():
    vectors = [
        np.array([2, 1]),
        np.array([1, 3]),
        np.array([3, 2]),
        np.array([-2, 1]),
        np.array([-1, -3])
    ]

    plot_vectors(
        vectors,
        save_path="outputs/vectors.png",
        show=True
    )

if __name__ == "__main__":
    run()
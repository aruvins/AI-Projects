import numpy as np
from visualizers.transformations import apply_transform, plot_transform

def run():
    vectors = [
        np.array([1, 2]),
        np.array([2, 1]),
        np.array([-1, 3])
    ]

    matrix = np.array([
        [2, 0],
        [0, 1]
    ])

    transformed = apply_transform(matrix, vectors)
    plot_transform(vectors, transformed)

if __name__ == "__main__":
    run()
import numpy as np
from visualizers.transformations import apply_transform, plot_transform

def run():
    vectors = [
        np.array([2, 1]),
        np.array([1, 3]),
        np.array([3, 2]),
        np.array([-2, 1]),
        np.array([-1, -3])
    ]

    matrix = np.array([
        [1.5, 0.5],
        [0.5, 1.5]
    ])

    transformed = apply_transform(matrix, vectors)

    plot_transform(
        vectors,
        transformed,
        save_path="outputs/transform.png",
        show=True
    )

if __name__ == "__main__":
    run()
import numpy as np
from visualizers.eigen_demo import show_eigen

def run():
    matrix = np.array([
        [2, 1],
        [1, 2]
    ])

    show_eigen(matrix)

if __name__ == "__main__":
    run()
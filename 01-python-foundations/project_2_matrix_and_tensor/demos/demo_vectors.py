import numpy as np
from visualizers.vectors import show_vectors

def run():
    v1 = np.array([2, 3])
    v2 = np.array([-1, 4])
    v3 = np.array([3, -2])

    show_vectors([v1, v2, v3])

if __name__ == "__main__":
    run()
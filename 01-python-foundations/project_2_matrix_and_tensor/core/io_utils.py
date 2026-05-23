import os
import matplotlib.pyplot as plt

def save_plot(path):
    """
    Save current matplotlib figure and close it.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
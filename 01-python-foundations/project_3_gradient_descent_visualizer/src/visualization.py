import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

COLORS = ["blue", "green", "red", "purple", "orange"]

# ------------------------------------------------
# Setup the plot
# ------------------------------------------------
def setup_plot(func):
    """
    Sets up the plot for visualizing the function and optimization paths.

    Parameters:
    - func: The function to visualize.

    Returns:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axes object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a range of x values for plotting the function
    x_values = np.linspace(-10, 10, 1000)
    y_values = func(x_values)

    # Plot the function
    ax.plot(x_values, y_values, linewidth=3, label="Function")

    ax.set_title("Gradient Descent Optimization")
    ax.set_xlabel("Parameter Value - (x)")
    ax.set_ylabel("Loss - f(x)")
    ax.legend()
    ax.grid()

    return fig, ax

# ------------------------------------------------
# Animate the optimization paths
# ------------------------------------------------
def animate_path(fig, ax, results):
    """
    Creates an animation of the optimization paths for different learning rates.

    Parameters:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axes object.
    - results: A list of tuples containing (learning_rate, path_x, path_y) for each optimization run.

    Returns:
    - ani: The FuncAnimation object for the animation.
    """

    points = []
    lines = []

    for i, (lr, path_x, path_y) in enumerate(results):
        color = COLORS[i % len(COLORS)]
        line, = ax.plot([], [], label=f"lr={lr}", color=color)
        point, = ax.plot([], [], "o", color=color)

        points.append((point, path_x, path_y))
        lines.append((line, path_x, path_y))

    ax.legend()
    max_frames = max(len(r[1]) for r in results)

    def update(frame):
        artists = []

        for (point, path_x, path_y), (line, _, _) in zip(points, lines):
            if frame < len(path_x):
                point.set_data([path_x[frame]], [path_y[frame]])
                line.set_data(
                    path_x[: frame + 1],
                    path_y[: frame + 1],
                )
            artists.extend([point, line])
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=300,
        blit=True,
        repeat=False,
    )

    return ani
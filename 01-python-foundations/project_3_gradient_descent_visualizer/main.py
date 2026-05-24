import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functions import quadratic_function, quadratic_gradient, wavy_function, wavy_gradient
from optimizer import gradient_descent
from visualization import setup_plot, animate_path

# ------------------------------------------------
# Configuration 
# ------------------------------------------------
START_X = 9.0
# START_X = -8.0
ITERATIONS = 40

LEARNING_RATES = [0.05, 0.2, 0.9] # Different learning rates to visualize
# LEARNING_RATES = [0.001, 0.1, 1.5]

# ------------------------------------------------
# Run Gradient Descent for each learning rate
# ------------------------------------------------
results = []

for lr in LEARNING_RATES:
    path_x, path_y = gradient_descent(
        # func = quadratic_function,
        func = wavy_function,
        # grad = quadratic_gradient,
        grad = wavy_gradient,
        start_x = START_X,
        learning_rate = lr,
        iterations = ITERATIONS
    )
    results.append((lr, path_x, path_y))

# ------------------------------------------------
# Visualization
# ------------------------------------------------
# fig, ax = setup_plot(quadratic_function)
fig, ax = setup_plot(wavy_function)

ani = animate_path(
    fig = fig,
    ax = ax,
    results = results
)

# Save animation
import os

os.makedirs("output", exist_ok=True)

output_path = os.path.join("output", "gradient_descent_visualization.gif")

ani.save(output_path, writer="pillow", fps=4)

print(f"Saved animation to: {output_path}")

plt.show()

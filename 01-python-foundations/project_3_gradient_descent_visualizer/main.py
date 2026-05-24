import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functions import quadratic_function, quadratic_gradient
from optimizer import gradient_descent
from visualization import setup_plot, animate_path

# ------------------------------------------------
# Configuration 
# ------------------------------------------------
START_X = 9.0
ITERATIONS = 40

LEARNING_RATES = [0.05, 0.2, 0.9] # Different learning rates to visualize

# ------------------------------------------------
# Run Gradient Descent for each learning rate
# ------------------------------------------------
results = []

for lr in LEARNING_RATES:
    path_x, path_y = gradient_descent(
        func = quadratic_function,
        grad = quadratic_gradient,
        start_x = START_X,
        learning_rate = lr,
        iterations = ITERATIONS
    )
    results.append((lr, path_x, path_y))

# ------------------------------------------------
# Visualization
# ------------------------------------------------
fig, ax = setup_plot(quadratic_function)

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

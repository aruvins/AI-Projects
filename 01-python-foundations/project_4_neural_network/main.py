import os

from src.data import generate_spiral_data
from src.network import NeuralNetwork
from src.visualization import (
    plot_loss,
    plot_accuracy,
    plot_predictions,
)


# -------------------------------------------------
# Create Output Folder
# -------------------------------------------------

os.makedirs("output", exist_ok=True)


# -------------------------------------------------
# Generate Dataset
# -------------------------------------------------

X, y = generate_spiral_data(
    points_per_class=200,
    num_classes=3,
)


# -------------------------------------------------
# Create Neural Network
# -------------------------------------------------

model = NeuralNetwork(
    input_size=2,
    hidden_size=256,
    output_size=3,
    learning_rate=0.2,
)


# -------------------------------------------------
# Train
# -------------------------------------------------

history = model.train(
    X,
    y,
    epochs=20000,
)


# -------------------------------------------------
# Save Visualizations
# -------------------------------------------------

plot_loss(history["loss"])
plot_accuracy(history["accuracy"])
plot_predictions(model, X, y)
print("Saved outputs to output/")
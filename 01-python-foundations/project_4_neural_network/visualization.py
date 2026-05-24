import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Loss Curve
# -------------------------------------------------


def plot_loss(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig("output/training_loss.png")



# -------------------------------------------------
# Accuracy Curve
# -------------------------------------------------


def plot_accuracy(accuracy_history):
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_history)

    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.savefig("output/training_accuracy.png")


# -------------------------------------------------
# Decision Boundary Visualization
# -------------------------------------------------


def plot_predictions(model, X, y):

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.01

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = model.forward(grid)

    Z = np.argmax(predictions, axis=1)

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, Z, alpha=0.4)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="k"
    )
    
    plt.title("Neural Network Decision Boundary")
    plt.savefig("output/predictions.png")

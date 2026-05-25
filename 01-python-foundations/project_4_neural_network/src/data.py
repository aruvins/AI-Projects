import numpy as np


# -------------------------------------------------
# Spiral Dataset
# -------------------------------------------------
# A classic neural network toy dataset
# -------------------------------------------------


def generate_spiral_data(points_per_class, num_classes):

    X = np.zeros((points_per_class * num_classes, 2))

    y = np.zeros(points_per_class * num_classes, dtype="uint8")

    for class_number in range(num_classes):

        ix = range(
            points_per_class * class_number,
            points_per_class * (class_number + 1),
        )

        radius = np.linspace(0.0, 1, points_per_class)

        theta = np.linspace(
            class_number * 4,
            (class_number + 1) * 4,
            points_per_class,
        )

        theta += np.random.randn(points_per_class) * 0.2

        X[ix] = np.c_[
            radius * np.sin(theta * 2.5),
            radius * np.cos(theta * 2.5),
        ]

        y[ix] = class_number

    return X, y
import numpy as np

def gradient_descent(func, grad, start_x, learning_rate, iterations):
    """
    Performs gradient descent optimization.

    Parameters:
    - func: The function to minimize.
    - grad: The gradient of the function. (Derivative of the function)
    - start_x: The starting point for optimization.
    - learning_rate: The step size for each iteration.
    - iterations: The number of optimization steps to perform.

    Returns:
    - path_x: List of x values visited during optimization.
    - path_y: List of corresponding function values at those x values.
    """
    x = start_x

    path_x = [x]
    path_y = [func(x)]

    for _ in range(iterations):
        gradient = grad(x)

        # Update x by moving against the gradient
        x = x - learning_rate * gradient

        path_x.append(x)
        path_y.append(func(x))

    return np.array(path_x), np.array(path_y)
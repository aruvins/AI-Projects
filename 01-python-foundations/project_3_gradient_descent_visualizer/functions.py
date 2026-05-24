import numpy as np

# ------------------------------------------------
# Function 1: Simple Quadratic Bowl
# ------------------------------------------------
# Minimum at x = 3, f(3) = 2
# ------------------------------------------------
def quadratic_function(x):
    """
    A simple quadratic function: f(x) = (x - 3)^2 + 2
    This function has its minimum at x = 3, where f(3) = 2.
    """
    return (x - 3)**2 + 2

def quadratic_gradient(x):
    """
    The gradient of the quadratic function: f'(x) = 2 * (x - 3)
    This is the derivative of the quadratic function.
    """
    return 2 * (x - 3)

# ------------------------------------------------
# Optional Additional Functions
# ------------------------------------------------

def wavy_function(x):
    """
    A wavy function with multiple minima: f(x) = sin(x) + 0.1 * (x - 3)^2
    This function has a global minimum near x = 3, but also has local minima due to the sine component.
    """
    return np.sin(x) + 0.1 * (x - 3)**2


def wavy_gradient(x):
    """
    The gradient of the wavy function: f'(x) = cos(x) + 0.2 * (x - 3)
    This is the derivative of the wavy function.
    """
    return np.cos(x) + 0.2 * (x - 3)
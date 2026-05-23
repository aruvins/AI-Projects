import numpy as np

def shape_info(tensor):
    return {
        "shape": tensor.shape,
        "size": tensor.size,
        "dimensions": tensor.ndim
    }

def reshape(tensor, new_shape):
    return np.reshape(tensor, new_shape)

def broadcast_add(A, B):
    return A + B
import numpy as np

class ReLU:
    '''
    ReLU (Rectified Linear Unit) activation function.
    The ReLU function is defined as:
    f(x) = max(0, x)
    It outputs the input directly if it is positive; otherwise, it outputs zero.
    It is widely used in neural networks due to its simplicity and effectiveness in introducing non-linearity.
    
    Backward pass:
    The gradient of the ReLU function is 1 for positive input values and 0 for non-positive input values. 
    During backpropagation, the gradient is passed through unchanged for positive inputs, and set to zero for non-positive inputs.
    '''
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.input <= 0] = 0
        return grad


class Softmax:
    '''
    Softmax activation function.
    The Softmax function is defined as:
    f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    It converts a vector of real numbers into a probability distribution.
    Each output value is in the range (0, 1), and the sum of all output values is 1.
    It is commonly used in the output layer of classification neural networks to represent class probabilities.
    '''
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )

        return probabilities
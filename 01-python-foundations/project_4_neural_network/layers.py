import numpy as np


class Dense:

    def __init__(self, input_size, output_size):
        self.weights = 0.01 * np.random.randn(
            input_size,
            output_size,
        )

        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        '''
        Dense (Fully Connected) Layer.
        This layer performs a linear transformation of the input data using weights and biases.
        
        The forward pass computes the output as:
        output = input * weights + biases
        
        The weights are initialized with small random values, and the biases are initialized to zero.
        '''

        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        '''
        Backward pass for the Dense layer.
        This method computes the gradients of the weights and biases with respect to the loss, and updates them using gradient descent.
        
        The gradients are calculated as follows:
        grad_weights = input^T * grad_output
        grad_biases = sum(grad_output)
        grad_input = grad_output * weights^T

        The weights and biases are then updated by subtracting the product of the learning rate and their respective gradients.
        '''

        grad_weights = np.dot(self.input.T, grad_output)
        
        grad_biases = np.sum(
            grad_output,
            axis=0,
            keepdims=True,
        )

        grad_input = np.dot(
            grad_output,
            self.weights.T,
        )

        # Gradient descent updates
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
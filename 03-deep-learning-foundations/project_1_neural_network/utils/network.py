import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Input -> Hidden Layer 
        self.W1 = np.random.rand(2, 4)  # Weights for the first layer (2 inputs, 4 neurons)
        self.b1 = np.zeros((1, 4))  # Biases for the first layer

        # Hidden Layer -> Output Layer
        self.W2 = np.random.rand(4, 1)  # Weights for the second layer (4 neurons, 1 output)
        self.b2 = np.zeros((1, 1))  # Biases for the output layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass through the network

        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear transformation
        self.a1 = self.sigmoid(self.z1)  # Activation function

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear transformation
        self.a2 = self.sigmoid(self.z2)  # Activation function

        return self.a2
    
    def loss(self, y_true, y_pred):
        # Mean Squared Error Loss
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y, learning_rate=0.1):
        # Backpropagation to update weights and biases

        # Output layer error
        output_error = y - self.a2  # Error at the output layer
        output_delta = output_error * self.sigmoid_derivative(self.a2)  # Delta for the output layer

        # Hidden layer error
        hidden_error = output_delta.dot(self.W2.T)  # Error at the hidden layer
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)  # Delta for the hidden layer

        # Update weights and biases
        self.W2 += self.a1.T.dot(output_delta) * learning_rate  # Update weights for the output layer
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate  # Update biases for the output layer

        self.W1 += X.T.dot(hidden_delta) * learning_rate  # Update weights for the hidden layer
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate  # Update biases for the hidden layer
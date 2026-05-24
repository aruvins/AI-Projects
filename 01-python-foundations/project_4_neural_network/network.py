import numpy as np

from layers import Dense
from activations import ReLU, Softmax
from losses import CrossEntropyLoss


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate,
    ):

        self.learning_rate = learning_rate

        self.layer1 = Dense(input_size, hidden_size)
        self.activation1 = ReLU()

        self.layer2 = Dense(hidden_size, 128)
        self.activation2 = ReLU()

        self.layer3 = Dense(128, 64)
        self.activation3 = ReLU()

        self.layer4 = Dense(64, output_size)
        self.softmax = Softmax()

        self.loss_function = CrossEntropyLoss()

    def forward(self, X):
        output1 = self.layer1.forward(X)
        activated1 = self.activation1.forward(output1)
        
        output2 = self.layer2.forward(activated1)
        activated2 = self.activation2.forward(output2)

        output3 = self.layer3.forward(activated2)
        activated3 = self.activation3.forward(output3)

        output4 = self.layer4.forward(activated3)
        probabilities = self.softmax.forward(output4)

        return probabilities

    def calculate_accuracy(self, predictions, y):
        predicted_classes = np.argmax(predictions, axis=1)
        return np.mean(predicted_classes == y)

    def train(self, X, y, epochs):
        history = {
            "loss": [],
            "accuracy": [],
        }

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute loss
            loss = self.loss_function.forward(predictions, y)

            # Compute accuracy
            accuracy = self.calculate_accuracy(predictions, y)

            # Backward pass
            grad_loss = self.loss_function.backward(predictions, y)
            grad_layer4 = self.layer4.backward(
                grad_loss,
                self.learning_rate,
            )

            grad_activation3 = self.activation3.backward(
                grad_layer4
            )
            grad_layer3 = self.layer3.backward(
                grad_activation3,
                self.learning_rate,
            )

            grad_activation2 = self.activation2.backward(
                grad_layer3
            )
            grad_layer2 = self.layer2.backward(
                grad_activation2,
                self.learning_rate,
            )

            grad_activation1 = self.activation1.backward(
                grad_layer2,
            )
            self.layer1.backward(
                grad_activation1,
                self.learning_rate,
            )

            history["loss"].append(loss)
            history["accuracy"].append(accuracy)
        return history
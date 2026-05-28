import numpy as np
import matplotlib.pyplot as plt
import os

from utils.network import NeuralNetwork

os.makedirs("outputs", exist_ok=True)

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])  # Input data

y = np.array([
    [0],
    [1],
    [1],
    [0]
])  # Output labels

nn = NeuralNetwork()

epochs = 10000

losses = []

for epoch in range(epochs):
    # Forward pass
    output = nn.forward(X)

    # Calculate loss
    loss = nn.loss(y, output)
    losses.append(loss)

    # Backward pass and update weights
    nn.backward(X, y)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Final output after training:")
print(nn.forward(X))

# ------------------------------------------------
# Plot the loss over epochs
# ------------------------------------------------
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("outputs/training_loss.png")
plt.show()

# ------------------------------------------------
# Visualize the decision boundary
# ------------------------------------------------
predictions = nn.forward(X)
plt.bar(["[0,0]", "[0,1]", "[1,0]", "[1,1]"],
    predictions.flatten()
)

plt.ylim(0, 1)
plt.xlabel("Input")
plt.ylabel("Predicted Output")
plt.title("Predicted Output for XOR Inputs")
plt.savefig("outputs/xor_predictions.png")
plt.show()

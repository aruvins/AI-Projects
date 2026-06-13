import json
import numpy as np
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import torch
import torch.nn as nn
import torch.optim as optim

from utils.bag_of_words import tokenize
from utils.bag_of_words import stem
from utils.bag_of_words import bag_of_words

from utils.model import ChatbotNN

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "data", "intents.json")

with open(data_path, "r") as f:
    intents = json.load(f)

nltk.download("punkt")

# -----------------------
# Load dataset
# -----------------------

with open(
    data_path,
    "r"
) as file:

    intents = json.load(file)

all_words = []
tags = []
xy = []

# -----------------------
# Build vocabulary
# -----------------------

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append(
            (tokens, tag)
        )

ignore_words = [
    "?",
    ".",
    ",",
    "!"
]

all_words = [stem(word) for word in all_words if word not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# -----------------------
# Create training data
# -----------------------

X_train = []
y_train = []

for pattern_sentence, tag in xy:
    bow = bag_of_words(
        pattern_sentence,
        all_words
    )

    X_train.append(bow)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# -----------------------
# Convert to tensors
# -----------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

# -----------------------
# Hyperparameters
# -----------------------

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
epochs = 1000

# -----------------------
# Model
# -----------------------

model = ChatbotNN(
    input_size,
    hidden_size,
    output_size
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# -----------------------
# Training Loop
# -----------------------

for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(
        outputs,
        y_train
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )

print("\nTraining Complete")

# -----------------------
# Save Model
# -----------------------

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(
    data,
    "chatbot_model.pth"
)

print(
    "Model saved to chatbot_model.pth"
)
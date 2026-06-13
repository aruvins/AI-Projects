import torch
import torch.nn as nn

class ChatbotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.net= nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.net(x)
        return x
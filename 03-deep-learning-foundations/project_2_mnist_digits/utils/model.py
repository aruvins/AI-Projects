import torch
import torch.nn as nn


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            
            # 28 x 28 = 784 input features, 128 output features
            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            # 10 output features (for 0 - 9 digits)
            nn.Linear(64, 10)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
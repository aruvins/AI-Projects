from typing import Any

import torch.nn as nn

class FashionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.network(x)
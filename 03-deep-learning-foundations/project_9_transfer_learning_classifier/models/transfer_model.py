import torch.nn as nn
from torchvision import models

class TransferClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features

        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
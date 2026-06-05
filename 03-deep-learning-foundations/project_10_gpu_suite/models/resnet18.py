import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes=10):
    # Load the ResNet-18 model without pretrained weights
    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
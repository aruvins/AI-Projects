import torch.nn as nn
from transformers import ViTForImageClassification


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )

        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(
            pixel_values=x
        ).logits
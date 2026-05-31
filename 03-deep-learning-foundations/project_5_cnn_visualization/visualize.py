import torch

from utils.model import CNN
from utils.dataset import get_dataloaders
from utils.feature_visualizer import visualize_feature_maps
from utils.filter_visualizer import visualize_filters
from utils.prediction_visualizer import visualize_predictions
from utils.activation_visualizer import visualize_activation_map

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = CNN().to(DEVICE)
model.load_state_dict(
    torch.load(
        "outputs/cnn_model.pth",
        map_location=DEVICE
    )
)

model.eval()

train_loader, _ = get_dataloaders()

images, labels = next(iter(train_loader))
sample_image = images[0]

import torch.nn as nn

conv_count = 1

for layer in model.features:
    if isinstance(layer,nn.Conv2d):
        visualize_filters(layer,f"conv{conv_count}_filters.png")
        conv_count += 1

visualize_feature_maps(model,sample_image,DEVICE)
visualize_predictions(model,images,labels,DEVICE)
visualize_activation_map(model,sample_image,DEVICE)
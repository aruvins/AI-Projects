import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import time
import torch

from models.cnn_model import ResNetClassifier
from models.vit_model import ViTClassifier


# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


cnn = ResNetClassifier().to(DEVICE)
vit = ViTClassifier().to(DEVICE)

dummy = torch.randn(
    1,
    3,
    224,
    224
).to(DEVICE)

# CNN benchmark
start = time.time()
cnn(dummy)
cnn_time = time.time() - start


# ViT benchmark
start = time.time()
vit(dummy)
vit_time = time.time() - start


print(f"CNN Inference Time: {cnn_time:.4f}s")
print(f"ViT Inference Time: {vit_time:.4f}s")
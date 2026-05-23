import os
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def create_directory(path):
    os.makedirs(path, exist_ok=True)



def save_image(image, path):
    image.save(path)



def show_image(path):
    image = Image.open
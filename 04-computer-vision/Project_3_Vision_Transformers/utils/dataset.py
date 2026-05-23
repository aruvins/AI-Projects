import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


train_dataset = CIFAR10(
    root="data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = CIFAR10(
    root="data",
    train=False,
    transform=transform,
    download=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
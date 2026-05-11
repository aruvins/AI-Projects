import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.Normalize(mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
])
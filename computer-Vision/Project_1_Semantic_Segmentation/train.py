import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from models.unet import UNet
from utils.transforms import train_transform
from utils.metrics import dice_score, iou_score


if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20


class PetSegmentationDataset(Dataset):
    def __init__(self, split="trainval", transform=None):
        self.dataset = OxfordIIITPet(
            root="data",
            split=split,
            target_types="segmentation",
            download=True
        )

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        image = np.array(image)
        mask = np.array(mask)

        # Convert trimap into binary mask
        mask = (mask == 1).astype(np.float32)

        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


train_dataset = PetSegmentationDataset(
    split="trainval",
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


model = UNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(NUM_EPOCHS):

    model.train()

    loop = tqdm(train_loader)

    total_loss = 0

    for images, masks in loop:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}]"
        )

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)

    print(f"Average Loss: {avg_loss:.4f}")

    # Validation metrics
    model.eval()

    total_dice = 0
    total_iou = 0

    with torch.no_grad():

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            total_dice += dice_score(preds, masks)
            total_iou += iou_score(preds, masks)

    print(f"Dice Score: {total_dice / len(train_loader):.4f}")
    print(f"IoU Score: {total_iou / len(train_loader):.4f}")


torch.save(model.state_dict(), "unet_model.pth")

print("Model saved successfully.")
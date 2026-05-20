
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMAGE_SIZE


class CaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, image_size=IMAGE_SIZE):
        self.image_dir = image_dir
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        with open(captions_file, "r") as f:
            for line in f:
                image_name, caption = line.strip().split("|")
                self.samples.append((image_name, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "caption": caption
        }
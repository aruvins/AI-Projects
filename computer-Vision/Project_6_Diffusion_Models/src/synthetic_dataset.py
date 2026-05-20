import os
from diffusers import StableDiffusionPipeline
import torch
from config import DEVICE

MODEL_ID = "runwayml/stable-diffusion-v1-5"


class SyntheticDatasetGenerator:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )

        self.pipe = self.pipe.to(DEVICE)

    def generate_dataset(self, prompts, output_dir="outputs/generated"):
        os.makedirs(output_dir, exist_ok=True)

        for idx, prompt in enumerate(prompts):
            image = self.pipe(prompt).images[0]

            filename = f"sample_{idx}.png"
            save_path = os.path.join(output_dir, filename)

            image.save(save_path)

            print(f"Saved {save_path}")


if __name__ == "__main__":
    prompts = [
        "Aerial view of a city",
        "Medical imaging scan",
        "Autonomous driving street scene",
        "Industrial machinery"
    ]

    generator = SyntheticDatasetGenerator()
    generator.generate_dataset(prompts)
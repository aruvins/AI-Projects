from diffusers import StableDiffusionPipeline
import torch
from config import MODEL_ID, DEVICE
import os

def generate_image(prompt, output_path="outputs/images/generated.png"):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32
    )

    pipe = pipe.to(DEVICE)

    image = pipe(prompt).images[0]

    image.save(output_path)

    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    os.makedirs("outputs/images", exist_ok=True)
    prompt = "A futuristic cyberpunk city at night"
    generate_image(prompt)
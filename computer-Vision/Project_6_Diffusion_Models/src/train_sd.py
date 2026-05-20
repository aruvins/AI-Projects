from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
import torch
from config import MODEL_ID, DEVICE

MODEL_ID = MODEL_ID


def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )

    pipe = pipe.to(DEVICE)

    return pipe


if __name__ == "__main__":
    pipe = load_model()

    prompt = "A fantasy castle on a mountain"

    image = pipe(prompt).images[0]
    image.save("outputs/images/fine_tuned_example.png")
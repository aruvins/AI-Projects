import os
import torch

from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from config import DEVICE, MODEL_ID


class ImageEditor:
    def __init__(
        self,
        model_id=MODEL_ID,
        device=DEVICE
    ):
        self.device = device

        dtype = torch.float32 if device == "mps" else torch.float16

        print("Loading inpainting pipeline...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )

        self.pipe = self.pipe.to(device)

        print("Pipeline loaded successfully.")

    def edit_image(
        self,
        image_path,
        mask_path,
        prompt,
        negative_prompt=None,
        strength=0.99,
        guidance_scale=7.5,
        num_inference_steps=50,
        output_path="outputs/edits/edited.png"
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Loading image: {image_path}")
        print(f"Loading mask: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        print("Generating edited image...")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )

        edited_image = result.images[0]

        edited_image.save(output_path)

        print(f"Edited image saved to: {output_path}")

        return edited_image


if __name__ == "__main__":
    editor = ImageEditor()

    editor.edit_image(
        image_path="data/input.jpg",
        mask_path="data/mask.png",
        prompt="Make the oval purple",
        negative_prompt="blurry, low quality"
    )
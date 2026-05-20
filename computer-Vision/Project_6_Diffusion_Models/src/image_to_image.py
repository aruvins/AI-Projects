from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
from config import DEVICE, MODEL_ID, IMAGE

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to(DEVICE)

init_image = Image.open(IMAGE).convert("RGB")

result = pipe(
    prompt="Turn this into a watercolor painting",
    image=init_image,
    strength=0.75,
    guidance_scale=7.5
).images[0]

result.save("outputs/images/watercolor.png")
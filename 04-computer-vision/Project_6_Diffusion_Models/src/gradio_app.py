import torch
import gradio as gr
from config import DEVICE, MODEL_ID

from diffusers import StableDiffusionPipeline


print(f"Using device: {DEVICE}")


class DiffusionApp:
    def __init__(self):
        dtype = torch.float32

        print("Loading Stable Diffusion model...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None
        )

        self.pipe = self.pipe.to(DEVICE)

        print("Model loaded successfully.")

    def generate_image(
        self,
        prompt,
        negative_prompt,
        guidance_scale,
        steps
    ):
        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps
            )

            image = result.images[0]

            return image

        except Exception as e:
            print("Generation Error:", e)
            raise e


app = DiffusionApp()


interface = gr.Interface(
    fn=app.generate_image,

    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="A futuristic cyberpunk city"
        ),

        gr.Textbox(
            label="Negative Prompt",
            placeholder="blurry, low quality"
        ),

        gr.Slider(
            minimum=1,
            maximum=20,
            value=7.5,
            step=0.5,
            label="Guidance Scale"
        ),

        gr.Slider(
            minimum=10,
            maximum=50,
            value=25,
            step=1,
            label="Inference Steps"
        )
    ],

    outputs=gr.Image(),

    title="Stable Diffusion Demo",

    description="Generate AI images with Stable Diffusion"
)


if __name__ == "__main__":
    interface.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860
    )
# This environment variable can help prevent issues with model downloads on certain systems.
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from controlnet_aux import OpenposeDetector
import gradio as gr

# --- 1. Load All Models (This runs only once when the app starts) ---

print("Loading all models. This may take a moment...")

# Load the OpenPose Detector
print("Loading OpenPose Detector...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# Load ControlNet
print("Loading ControlNet model...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)

# Load the base Stable Diffusion pipeline
print("Loading Stable Diffusion 1.5 pipeline...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Load and configure the IP-Adapter
print("Loading IP-Adapter model...")
ip_adapter_model_id = "h94/IP-Adapter"
pipe.load_ip_adapter(ip_adapter_model_id, subfolder="models", weight_name="ip-adapter_sd15.bin")

# Configure the pipeline for performance and quality
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xFormers memory efficient attention enabled.")
except ImportError:
    print("xFormers is not installed. For faster inference, consider installing it with: pip install xformers")

print("All models loaded successfully!")



# --- 2. Define the Main Inference Function ---
# This function will be called by Gradio every time the user clicks "Generate"

def generate_image(
    source_person_img,
    style_ref_img,
    prompt,
    negative_prompt,
    ip_adapter_scale,
    guidance_scale,
    num_steps,
    seed
):
    """
    Generates an image based on a source person, a style reference, and a text prompt.
    """
    if source_person_img is None or style_ref_img is None:
        raise gr.Error("Please upload both a source image for the pose and a style reference image.")

    # Convert Gradio's numpy array inputs to PIL Images
    source_image = Image.fromarray(source_person_img)
    style_image = Image.fromarray(style_ref_img)
    
    # 1. Detect pose from the source image
    print("Detecting pose...")
    pose_image = openpose(source_image)

    # 2. Set the IP-Adapter scale for this run
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    # 3. Set the generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 4. Run the full pipeline
    print("Generating image...")
    output_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pose_image,
        ip_adapter_image=style_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    print("Generation complete.")
    return output_image

   
# --- 3. Create the Gradio Web Interface ---

with gr.Blocks(css="body {background-color: #f0f0f0;}") as app:
    gr.Markdown("# Virtual Try-On with ControlNet + IP-Adapter")
    gr.Markdown("Upload an image of a person (for the pose) and a style/texture image (for the clothes). The model will then generate a new image combining the pose and the style.")

    with gr.Row():
        with gr.Column():
            source_person_input = gr.Image(type="numpy", label="Source Image (for Pose)")
            style_ref_input = gr.Image(type="numpy", label="Style Reference Image (for Clothes)")
            generate_btn = gr.Button("Generate Image", variant="primary")
            gr.Markdown("### Example Images")
            gr.Examples(
                examples=[
                    ["source_person.png", "style_image.png"]
                ],
                inputs=[source_person_input, style_ref_input]
            )

        with gr.Column():
            output_image = gr.Image(type="pil", label="Generated Output")

    with gr.Accordion("Advanced Settings", open=False):
        prompt_input = gr.Textbox(label="Prompt", value="masterpiece, best quality, a woman in a beautiful dress, photorealistic, fashion photo")
        neg_prompt_input = gr.Textbox(label="Negative Prompt", value="monochrome, lowres, bad anatomy, worst quality, jpeg artifacts, blurry, bad hands")
        ip_adapter_scale_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.6, label="Style Strength (IP-Adapter Scale)")
        guidance_scale_slider = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, value=7.5, label="Prompt Strength (CFG Scale)")
        steps_slider = gr.Slider(minimum=20, maximum=100, step=1, value=40, label="Inference Steps")
        seed_input = gr.Number(label="Seed", value=12345, precision=0)

    generate_btn.click(
        fn=generate_image,
        inputs=[
            source_person_input,
            style_ref_input,
            prompt_input,
            neg_prompt_input,
            ip_adapter_scale_slider,
            guidance_scale_slider,
            steps_slider,
            seed_input
        ],
        outputs=output_image
    )

# To run this app, save it as a Python file (e.g., app.py) and run `gradio app.py` in your terminal.
if __name__ == "__main__":
    app.launch()

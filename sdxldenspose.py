# This environment variable can help prevent issues with model downloads on certain systems.
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
import cv2
import numpy as np
from PIL import Image
import random

# --- Check for Dependencies and Import ---
# This section checks for necessary libraries and provides helpful error messages.
try:
    import gradio as gr
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLImg2ImgPipeline,
        ControlNetModel,
        AutoencoderKL,
        DPMSolverMultistepScheduler
    )
except ImportError:
    print("="*80)
    print("ERROR: Diffusers or Gradio not found.")
    print("Please install the required libraries: pip install diffusers transformers accelerate gradio")
    print("="*80)
    exit()

try:
    from densepose import add_densepose_config
    from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
    from densepose.vis.extractor import DensePoseResultExtractor
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
except ImportError:
    print("="*80)
    print("ERROR: Failed to import Detectron2 or DensePose.")
    print("This application requires Detectron2 and DensePose for the control signal.")
    print("Please follow their official installation instructions carefully.")
    print("- PyTorch: https://pytorch.org/")
    print("- Detectron2: https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md")
    print("="*80)
    exit()

# --- 1. DensePose Predictor Class ---
# This class handles the DensePose segmentation.

class DensePosePredictor:
    """A wrapper class for running DensePose inference."""
    def __init__(self, config_path, weights_path):
        """Initializes the DensePose predictor."""
        if not all(os.path.exists(p) for p in [config_path, weights_path]):
            raise FileNotFoundError(
                f"Could not find DensePose model files. "
                f"Checked for config: '{config_path}' and weights: '{weights_path}'. "
                "Please update the 'densepose_ckpt_dir' variable in the script."
            )
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.predictor = DefaultPredictor(self.cfg)
        self.extractor = DensePoseResultExtractor()
        self.visualizer = Visualizer()

    def predict_seg(self, image_bgr):
        """Generates a DensePose segmentation mask for the input image."""
        with torch.no_grad():
            outputs = self.predictor(image_bgr)["instances"]
        outputs = self.extractor(outputs)
        image_seg = np.zeros(image_bgr.shape, dtype=np.uint8)
        self.visualizer.visualize(image_seg, outputs)
        return image_seg

# --- 2. Setup & Load All Models (Runs once on startup) ---

print("Loading all models. This is a large pipeline and may take a significant amount of time...")

# --- Model Paths ---
# IMPORTANT: Update this path to your local directory containing DensePose models.
densepose_ckpt_dir = "./ckpts" 
densepose_config_path = os.path.join(densepose_ckpt_dir, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml")
densepose_weights_path = os.path.join(densepose_ckpt_dir, "densepose", "model_final_162be9.pkl")

# Load DensePose Predictor
print("Loading DensePose Predictor...")
densepose_predictor = DensePosePredictor(
    config_path=densepose_config_path,
    weights_path=densepose_weights_path,
)
print("DensePose loaded.")

# Load the SDXL-compatible ControlNet for DensePose
print("Loading ControlNet model for SDXL DensePose...")
controlnet = ControlNetModel.from_pretrained(
    "jschoormans/controlnet-densepose-sdxl", 
    torch_dtype=torch.float16
)
print("ControlNet loaded.")

# Load the superior VAE made for SDXL
print("Loading high-quality VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)
print("VAE loaded.")

# Load the base SDXL pipeline with ControlNet and the custom VAE
print("Loading Base SDXL ControlNet Pipeline...")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
print("Base pipeline loaded.")

# Load the SDXL refiner pipeline, sharing components to save VRAM
print("Loading SDXL Refiner Pipeline...")
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
print("Refiner pipeline loaded.")


# Load and configure the SDXL IP-Adapter
print("Loading IP-Adapter for SDXL...")
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin" # Using the standard SDXL IP-Adapter to fix shape mismatch
)
print("IP-Adapter loaded.")

# Configure performance optimizations
# Using model CPU offloading is recommended for systems with limited VRAM
print("Configuring performance settings...")
pipe.enable_model_cpu_offload()
refiner.enable_model_cpu_offload()
print("CPU offloading enabled for base and refiner models.")

print("--- All models loaded successfully! ---")


# --- 3. Define the Main Inference Function ---

def generate_image(
    source_person_img, style_ref_img,
    prompt, negative_prompt,
    ip_adapter_scale, guidance_scale, num_steps, seed,
    width, height, clip_skip, refiner_strength
):
    """
    Generates an image using the full Base + Refiner + ControlNet + IP-Adapter pipeline.
    """
    if source_person_img is None or style_ref_img is None:
        raise gr.Error("Please upload both a source image for the pose and a style reference image.")

    # Seed handling for reproducibility
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Convert Gradio's numpy RGB inputs
    source_image_bgr = cv2.cvtColor(source_person_img, cv2.COLOR_RGB2BGR)
    style_image_pil = Image.fromarray(style_ref_img)
    
    # 1. Detect DensePose from the source image
    print("Detecting DensePose...")
    densepose_bgr = densepose_predictor.predict_seg(source_image_bgr)
    densepose_image = Image.fromarray(densepose_bgr[:, :, ::-1])

    # 2. Set the IP-Adapter scale
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    # 3. Base Model Pass (generates latent image)
    print(f"Running base model... (Seed: {seed})")
    latent_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=densepose_image,
        ip_adapter_image=style_image_pil,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        clip_skip=clip_skip,
        denoising_end=refiner_strength, # Stop before the end for the refiner
        output_type='latent'            # Output latents to pass to the refiner
    ).images[0]
    
    print("Base model pass complete. Running refiner...")

    # 4. Refiner Model Pass (adds fine details)
    final_image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        image=latent_image,             # Pass the latents from the base model
        denoising_start=refiner_strength # Start from where the base model left off
    ).images[0]
    
    print("Generation complete.")
    return final_image, densepose_image, seed


# --- 4. Create the Gradio Web Interface ---

with gr.Blocks(css="body {background-color: #f4f4f5;}") as app:
    gr.Markdown("# Ultimate SDXL Virtual Try-On")
    gr.Markdown("Combines **DensePose ControlNet**, **IP-Adapter**, a **high-quality VAE**, and a **Refiner** for state-of-the-art results.")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                source_person_input = gr.Image(type="numpy", label="Source Image (for Pose)")
                style_ref_input = gr.Image(type="numpy", label="Style Reference (for Clothes/Texture)")
            
            prompt_input = gr.Textbox(label="Prompt", value="a fashion model, high resolution, detailed, professional photo, 4k", lines=3)
            neg_prompt_input = gr.Textbox(label="Negative Prompt", value="low quality, worst quality, bad anatomy, bad hands, blurry, extra limbs, deformed, text, watermark", lines=2)
            
            with gr.Accordion("Advanced Settings", open=True):
                 with gr.Row():
                    width_slider = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                    height_slider = gr.Slider(512, 1536, value=1024, step=64, label="Height")
                 
                 steps_slider = gr.Slider(10, 100, value=40, step=1, label="Inference Steps")
                 guidance_slider = gr.Slider(1.0, 15.0, value=6.0, step=0.1, label="Guidance Scale (CFG)")
                 ip_adapter_scale_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="Style Strength (IP-Adapter Scale)")
                 refiner_strength_slider = gr.Slider(0.7, 1.0, value=0.8, step=0.05, label="Refiner Strength")
                 clip_skip_slider = gr.Slider(0, 4, value=0, step=1, label="CLIP Skip")
                 
                 with gr.Row():
                     seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                     random_seed_button = gr.Button("🎲")

            generate_button = gr.Button("Generate Image ✨", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Result", type="pil")
            densepose_output_vis = gr.Image(label="DensePose Control Map", type="pil")
            seed_output = gr.Number(label="Used Seed", interactive=False)
            
    # --- UI Event Handlers ---
    inputs = [
        source_person_input, style_ref_input,
        prompt_input, neg_prompt_input,
        ip_adapter_scale_slider, guidance_slider, steps_slider, seed_input,
        width_slider, height_slider, clip_skip_slider, refiner_strength_slider
    ]
    outputs = [output_image, densepose_output_vis, seed_output]
    
    generate_button.click(fn=generate_image, inputs=inputs, outputs=outputs)
    random_seed_button.click(lambda: -1, inputs=[], outputs=seed_input)

if __name__ == "__main__":
    app.launch(debug=True)

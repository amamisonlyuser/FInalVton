import torch
from PIL import Image, ImageFilter
import numpy as np
import gradio as gr
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from diffusers.utils import load_image
import os
import cv2

# Assuming mask.py and leffa_utils are accessible
from mask import HumanParsingMaskGenerator
from leffa_utils.utils import preserve_face_and_hair

# --- Label Map for Human Parsing ---
label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9,
    "right_shoe": 10, "head": 11, "left_leg": 12, "right_leg": 13,
    "left_arm": 14, "right_arm": 15, "bag": 16, "scarf": 17, "torso_skin": 18
}

# Global variables
pipeline = None
ip_adapter_loaded = False
mask_generator = None

# --- Model Initialization ---
def initialize_models():
    """
    Initializes all models, loading the main pipeline, a custom VAE,
    and the IP-Adapter from the Hugging Face Hub.
    """
    global pipeline, ip_adapter_loaded, mask_generator
    print("--- Initializing Models (This may take a moment) ---")

    # Hugging Face model IDs
    MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    VAE_ID = "madebyollin/sdxl-vae-fp16-fix"  # High-quality VAE
    IP_ADAPTER_ID = "h94/IP-Adapter"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Load the high-quality VAE
    try:
        print(f"Loading VAE from Hugging Face ID: {VAE_ID}")
        vae = AutoencoderKL.from_pretrained(
            VAE_ID,
            torch_dtype=torch_dtype
        )
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return False

    # 2. Load the pipeline with the custom VAE
    try:
        print(f"Loading Stable Diffusion XL Inpaint Pipeline from Hugging Face ID: {MODEL_ID}")
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,  # Pass the custom VAE here
            torch_dtype=torch_dtype,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
        )
    except Exception as e:
        print(f"Error loading pipeline from Hugging Face: {e}")
        return False

    # 3. Set the optimized scheduler
    print("Configuring DPMSolverMultistepScheduler with karras sigmas and sde-dpmsolver++.")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"  # Recommended for high quality
    )

    # 4. Move pipeline to the appropriate device
    try:
        pipeline.to(device)
        print(f"Pipeline moved to {device.upper()}.")
    except Exception as e:
        print(f"Error moving pipeline to device: {e}")
        return False

    # 5. Load IP-Adapter for SDXL
    try:
        print(f"Loading IP-Adapter from Hugging Face ID: {IP_ADAPTER_ID}")
        pipeline.load_ip_adapter(IP_ADAPTER_ID, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        ip_adapter_loaded = True
        print("IP-Adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading IP-Adapter: {e}")
        ip_adapter_loaded = False
        return False

    # 6. Initialize Mask Generator
    try:
        mask_generator = HumanParsingMaskGenerator()
        print("HumanParsingMaskGenerator initialized successfully.")
    except Exception as e:
        print(f"Error initializing HumanParsingMaskGenerator: {e}")
        return False

    print("--- Model Initialization Complete ---")
    return True

# --- Prediction Function for Gradio ---
def predict_inpaint(
    original_image_pil: Image.Image,
    ip_adapter_image_pil: Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    denoise_strength: float,
    resize_width: int,
    resize_height: int,
    # The mask smoothing/expansion is handled automatically by the mask generator now
):
    """
    Main function to run the inpainting workflow.
    """
    if pipeline is None or not ip_adapter_loaded or mask_generator is None:
        gr.Warning("Models not initialized or failed to load. Attempting re-initialization...")
        if not initialize_models():
            gr.Error("Model initialization failed. Cannot proceed.")
            return None, None

    if original_image_pil is None or ip_adapter_image_pil is None:
        gr.Warning("Please upload both an original image and an IP-Adapter reference image.")
        return None

    # Use a random seed if -1 is provided
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)
        
    original_image_pil = original_image_pil.convert("RGB")
    ip_adapter_image_pil = ip_adapter_image_pil.convert("RGB")

    # Recommended SDXL resolution for processing
    sdxl_input_width, sdxl_input_height = 1024, 1024

    # STEP 1: Generate the full parse map for creating the inpainting mask and for face preservation
    print("Generating segmentation mask...")
    model_parse = mask_generator.generate_mask(original_image_pil)
    if model_parse is None:
        gr.Error("Could not generate a human segmentation mask from the input image.")
        return None

    # STEP 2: Resize inputs and run the inpainting pipeline
    original_image_resized = original_image_pil.resize((sdxl_input_width, sdxl_input_height), Image.LANCZOS)
    # The mask is derived from the parsed map
    processed_mask_resized = model_parse.resize((sdxl_input_width, sdxl_input_height), Image.NEAREST)
    processed_mask_resized.save("processed_mask_resized.png")
    original_image_resized.save("original_image_resized.png")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))

    print("Running inpainting pipeline...")
    inpainted_images = pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=original_image_resized,
        mask_image=processed_mask_resized,
        ip_adapter_image=ip_adapter_image_pil,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
        strength=denoise_strength,
    ).images

    inpainted_image = inpainted_images[0]

    # STEP 3: Preserve the original face and hair
    print("Preserving original face and hair...")
    # Resize the generated image back to the original's dimensions for accurate pasting
    inpainted_image_orig_size = inpainted_image.resize((original_image_pil.width, original_image_pil.height), Image.LANCZOS)
    final_image = preserve_face_and_hair(original_image_pil, inpainted_image_orig_size, model_parse)

    # STEP 4: Final resize to user-specified output dimensions
    final_image = final_image.resize((resize_width, resize_height), Image.LANCZOS)

    return final_image


# --- Gradio Interface ---
with gr.Blocks(title="SDXL Inpainting Workflow") as demo:
    gr.Markdown(
        """
        # SDXL Inpainting Workflow 👩‍🎨
        This application uses an SDXL inpainting model to replace parts of an image.
        The mask is automatically generated to target the body, and the original face/hair are preserved.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            original_image_input = gr.Image(type="pil", label="1. Original Image")
            ip_adapter_image_input = gr.Image(type="pil", label="2. IP-Adapter Reference Image")
            positive_prompt_input = gr.Textbox(label="Positive Prompt", lines=3, value="body, skin, hands, torso, legs, shoulder, no cloths, only body parts, high quality, realistic")
            negative_prompt_input = gr.Textbox(label="Negative Prompt", lines=3, value="ugly, text, watermark, deformed, blurry, low resolution, cloths, dress, pants, shirt")

        with gr.Column(scale=1):
            gr.Markdown("### Generation Parameters")
            steps_input = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps")
            cfg_scale_input = gr.Slider(minimum=1.0, maximum=20.0, value=7.0, step=0.1, label="CFG Scale")
            denoise_strength_input = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Denoise Strength")
            seed_input = gr.Number(label="Seed (use -1 for random)", value=42, precision=0)

            gr.Markdown("### Output Settings")
            resize_width_input = gr.Number(label="Final Output Width", value=1024, precision=0)
            resize_height_input = gr.Number(label="Final Output Height", value=1024, precision=0)

            generate_button = gr.Button("Generate", variant="primary")

    with gr.Row():
        output_image = gr.Image(type="pil", label="Final Image")

    generate_button.click(
        fn=predict_inpaint,
        inputs=[
            original_image_input,
            ip_adapter_image_input,
            positive_prompt_input,
            negative_prompt_input,
            seed_input,
            steps_input,
            cfg_scale_input,
            denoise_strength_input,
            resize_width_input,
            resize_height_input,
            # Removed unused mask parameters
        ],
        outputs=output_image
    )

# Launch the Gradio app
if __name__ == "__main__":
    # Initialize models once when the script starts
    if not initialize_models():
        print("\nFATAL: Failed to initialize models. The application cannot start correctly.")
    else:
        # Launch the app if models loaded successfully
        demo.launch()
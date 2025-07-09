import torch
from PIL import Image, ImageFilter
import numpy as np
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import os

# --- Configuration ---
# Set the path to your Stable Diffusion checkpoint and IP-Adapter models
# Make sure these paths are correct for your system.
# The main model should be a Stable Diffusion 1.5 inpainting model.
CHECKPOINT_PATH = r"C:\\Users\\John123\\Documents\\ComfyUI\\models\\checkpoints\\lazymixRealAmateur_v40Inpainting.safetensors"
# IP_ADAPTER_MODEL_FULL_PATH should be the full path to the ip-adapter-plus_sd15.bin file.
IP_ADAPTER_MODEL_FULL_PATH = r"C:\Users\John123\Documents\ComfyUI\models\ipadapter"
IP_ADAPTER_IMAGE_ENCODER = r"C:\\Users\\John123\\Documents\\ComfyUI\\models\\clip_vision\\CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" # CLIP Vision model for IP-Adapter

# Global variables to store the loaded pipeline and IP-Adapter status
# These will be initialized once when the Gradio app starts
pipeline = None
ip_adapter_loaded = False # Flag to indicate if IP-Adapter has been successfully loaded

# --- Helper Functions for Mask Processing ---

def load_and_process_mask(mask_image_pil, smooth_amount, expand_size):
    """
    Processes a PIL mask image: converts to grayscale, smooths, and expands it.
    Mimics ComfyUI's ImageToMask, MaskSmooth+, and MaskExpandBatch+.
    """
    if mask_image_pil is None:
        raise ValueError("Mask image cannot be None.")

    # Convert to grayscale (L mode)
    mask = mask_image_pil.convert("L")

    # MaskSmooth+ (Approximation using Gaussian Blur)
    if smooth_amount > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=smooth_amount))

    # Convert to binary mask (0-255) if not already. Threshold at 128.
    mask = mask.point(lambda p: 255 if p > 128 else 0)

    # MaskExpandBatch+ (Approximation using Dilation)
    if expand_size > 0:
        # Create a kernel for dilation (e.g., a square kernel)
        kernel = ImageFilter.MaxFilter(size=expand_size * 2 + 1) # Size needs to be odd
        mask = mask.filter(kernel)

    return mask.convert("L") # Ensure it's L mode (grayscale) for diffusers

# --- Model Initialization ---
def initialize_models():
    global pipeline, ip_adapter_loaded
    print("--- Initializing Models (This may take a moment) ---")

    # 1. Load Checkpoint using from_single_file for .safetensors
    print(f"Loading Stable Diffusion Inpaint Pipeline from: {CHECKPOINT_PATH}")
    try:
        # Use from_single_file for direct .safetensors checkpoint loading
        pipeline = StableDiffusionInpaintPipeline.from_single_file(
            CHECKPOINT_PATH,
            torch_dtype=torch.float16,
            safety_checker=None, # Disable safety checker if not needed
            # For inpainting, ensure the model is compatible.
            # If from_single_file doesn't correctly identify it as inpainting,
            # you might need to load a base inpainting pipeline first and then load weights.
            # However, for most inpainting .safetensors, this should work.
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Ensure the checkpoint path is correct and it's a compatible Stable Diffusion Inpainting model.")
        print("If it's a .safetensors file, ensure `pip install safetensors` is done.")
        return False # Indicate failure

    # Set scheduler (Node 14: sampler_name="dpmpp_3m_sde", scheduler="karras")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
    
    # Move pipeline to GPU if available
    try:
        pipeline.to("cuda")
        print("Pipeline moved to CUDA (GPU).")
    except RuntimeError:
        print("CUDA not available, pipeline will run on CPU. This will be slower.")
        pipeline.to("cpu")


    # 2. Load IP-Adapter
    print(f"Loading IP-Adapter weights from: {IP_ADAPTER_MODEL_FULL_PATH}")
    print(f"Loading CLIP Vision model from: {IP_ADAPTER_IMAGE_ENCODER}")
    try:
        # Load the IP-Adapter into the pipeline by providing the full path to the .bin file
        # and the image encoder path.
        # subfolder and weight_name are not needed when the full path is given.
        ip_adapter_model_id = "h94/IP-Adapter"
        pipeline.load_ip_adapter(ip_adapter_model_id, subfolder="models", weight_name="ip-adapter_sd15.bin")
        ip_adapter_loaded = True # Set flag to True on successful load

    except Exception as e:
        print(f"Error loading IP-Adapter: {e}")
        print("Ensure the IP-Adapter model file and CLIP Vision model path are correct and compatible.")
        ip_adapter_loaded = False # Set flag to False on failure
        return False # Indicate failure

    print("--- Model Initialization Complete ---")
    return True # Indicate success

# --- Prediction Function for Gradio ---
def predict_inpaint(
    original_image_pil: Image.Image,
    ip_adapter_image_pil: Image.Image,
    mask_image_pil: Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    denoise_strength: float,
    resize_width: int,
    resize_height: int,
    mask_smooth_amount: int,
    mask_expand_size: int
):
    """
    Main function to run the inpainting workflow, adapted for Gradio.
    """
    if pipeline is None or not ip_adapter_loaded:
        gr.Warning("Models not initialized or IP-Adapter failed to load. Please wait or check console for errors.")
        if not initialize_models(): # Attempt to re-initialize if not loaded
            return None # Return None if initialization fails

    if original_image_pil is None:
        gr.Warning("Please upload an original image.")
        return None
    if ip_adapter_image_pil is None:
        gr.Warning("Please upload an IP-Adapter reference image.")
        return None
    if mask_image_pil is None:
        gr.Warning("Please upload an inpainting mask.")
        return None

    # Ensure images are in RGB mode
    original_image_pil = original_image_pil.convert("RGB")
    ip_adapter_image_pil = ip_adapter_image_pil.convert("RGB")

    # Process Mask
    processed_mask = load_and_process_mask(mask_image_pil, mask_smooth_amount, mask_expand_size)

    # Resize original image and mask
    original_image_resized = original_image_pil.resize((resize_width, resize_height), Image.LANCZOS)
    processed_mask_resized = processed_mask.resize((resize_width, resize_height), Image.NEAREST)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    print("Running inpainting pipeline...")
    # Call the pipeline directly, passing the ip_adapter_image
    images = pipeline(
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

    return images[0]

# --- Gradio Interface ---

# Initialize models when the script starts
if not initialize_models():
    print("Failed to initialize models. Gradio app may not function correctly.")

with gr.Blocks(title="ComfyUI Inpainting Workflow") as demo:
    gr.Markdown(
        """
        # ComfyUI Inpainting Workflow (Gradio App)
        This application converts a ComfyUI inpainting workflow into an interactive Gradio interface.
        Upload your images, set your prompts and parameters, and generate an inpainted image.
        """
    )

    with gr.Row():
        with gr.Column():
            original_image_input = gr.Image(type="pil", label="1. Original Image (Screenshot 2025-06-06 051538.png)", value="Screenshot 2025-06-06 051538.png")
            ip_adapter_image_input = gr.Image(type="pil", label="2. IP-Adapter Reference Image (image1 (3).png)", value="image1 (3).png")
            mask_image_input = gr.Image(type="pil", label="3. Inpainting Mask (skin_torso_mask.png)", value="skin_torso_mask.png")

            positive_prompt_input = gr.Textbox(label="Positive Prompt", lines=2, value="body, skin, hands, torso, legs, shoulder, no cloths, only body parts")
            negative_prompt_input = gr.Textbox(label="Negative Prompt", lines=2, value="ugly, text, watermark")

        with gr.Column():
            gr.Markdown("### KSampler Parameters")
            seed_input = gr.Number(label="Seed", value=39163971096256, precision=0)
            steps_input = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Steps")
            cfg_scale_input = gr.Slider(minimum=1.0, maximum=20.0, value=8.0, step=0.1, label="CFG Scale")
            denoise_strength_input = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Denoise Strength")

            gr.Markdown("### Image & Mask Processing")
            resize_width_input = gr.Number(label="Resize Width", value=768, precision=0)
            resize_height_input = gr.Number(label="Resize Height", value=1024, precision=0)
            mask_smooth_amount_input = gr.Slider(minimum=0, maximum=20, value=5, step=1, label="Mask Smooth Amount")
            mask_expand_size_input = gr.Slider(minimum=0, maximum=20, value=8, step=1, label="Mask Expand Pixels")

            generate_button = gr.Button("Generate Inpainted Image", variant="primary")

    with gr.Row():
        output_image = gr.Image(type="pil", label="Inpainted Image")

    generate_button.click(
        fn=predict_inpaint,
        inputs=[
            original_image_input,
            ip_adapter_image_input,
            mask_image_input,
            positive_prompt_input,
            negative_prompt_input,
            seed_input,
            steps_input,
            cfg_scale_input,
            denoise_strength_input,
            resize_width_input,
            resize_height_input,
            mask_smooth_amount_input,
            mask_expand_size_input
        ],
        outputs=output_image
    )

# Launch the Gradio app
if __name__ == "__main__":
    # Ensure you have the necessary libraries installed:
    # pip install torch diffusers transformers accelerate safetensors Pillow numpy gradio
    # Ensure you have a CUDA-compatible GPU for optimal performance.
    demo.launch()


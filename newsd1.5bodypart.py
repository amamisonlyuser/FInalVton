import torch
from PIL import Image, ImageFilter
import numpy as np
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import os
import cv2 # Added for the new function

# Assuming mask.py is in the same directory or accessible via PYTHONPATH
from mask import HumanParsingMaskGenerator
from leffa_utils.utils import  preserve_face_and_hair

# --- Configuration ---
# Set the path to your Stable Diffusion checkpoint and IP-Adapter models
# Make sure these paths are correct for your system.
# The main model should be a Stable Diffusion 1.5 inpainting model.
CHECKPOINT_PATH = r"C:\\Users\\John123\\Documents\\ComfyUI\\models\\checkpoints\\lazymixRealAmateur_v40Inpainting.safetensors"
# IP_ADAPTER_MODEL_FULL_PATH should be the full path to the ip-adapter-plus_sd15.bin file.
# Note: For `pipeline.load_ip_adapter`, you typically provide the model ID or a local directory
# where the model is organized in the diffusers format. If you have just the .bin file,
# you might need to structure it or point to a hub model.
# The current setup assumes "h94/IP-Adapter" is being used, and `weight_name` points to the specific file.
# So, IP_ADAPTER_MODEL_FULL_PATH as a directory containing the model's structure is more accurate.
# Let's adjust this to reflect diffusers' expectations.
# For a local file, it's often better to specify the base model ID and then the specific weight file.
IP_ADAPTER_MODEL_FULL_PATH = r"C:\Users\John123\Documents\ComfyUI\models\ipadapter"
IP_ADAPTER_IMAGE_ENCODER = r"C:\\Users\\John123\\Documents\\ComfyUI\\models\\clip_vision\\CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" # CLIP Vision model for IP-Adapter

# --- Label Map for Human Parsing ---
# This dictionary maps body parts to their corresponding integer values from the parsing model.
label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9,
    "right_shoe": 10, "head": 11, "left_leg": 12, "right_leg": 13,
    "left_arm": 14, "right_arm": 15, "bag": 16, "scarf": 17, "torso_skin": 18
}


# Global variables to store the loaded pipeline and IP-Adapter status
# These will be initialized once when the Gradio app starts
pipeline = None
ip_adapter_loaded = False # Flag to indicate if IP-Adapter has been successfully loaded

# Initialize mask generator globally
mask_generator = None

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

def generate_mask_from_image(src_image_pil, target_width, target_height, parts_to_mask=None, boundary_scale=1.0):
    """
    Generates a mask using the HumanParsingMaskGenerator.
    """
    global mask_generator
    if mask_generator is None:
        # Initialize mask_generator when first needed
        # It's better to initialize it once at the start, similar to the SD pipeline.
        # For now, keep it here for demonstration, but consider moving to initialize_models.
        try:
            mask_generator = HumanParsingMaskGenerator(ckpt_dir="./ckpts", device="cuda" if torch.cuda.is_available() else "cpu")
            print("HumanParsingMaskGenerator initialized.")
        except Exception as e:
            print(f"Error initializing HumanParsingMaskGenerator: {e}")
            return None

    if src_image_pil is None:
        return None

    # Default parts to mask as per the original request for 'skin'
    if parts_to_mask is None:
        parts_to_mask = [12, 13, 14, 15, 18] # torso, arms, legs, skin, face_skin

    try:
        # This function is assumed to return a binary mask for the specified parts.
        generated_mask_pil = mask_generator.generate_mask(
            src_image_pil=src_image_pil,
            target_labels=parts_to_mask,
            fill_holes=True,
        )
        return generated_mask_pil

    except Exception as e:
        print(f"Error during mask generation: {e}")
        return None

def generate_parse_map(src_image_pil):
    """
    Generates a full parsing map from the source image.
    This assumes the mask_generator has a method 'get_parsing_map' or similar.
    If not, this function would need to be adapted to the specific logic of HumanParsingMaskGenerator.
    """
    global mask_generator
    if mask_generator is None:
        gr.Warning("Mask generator not initialized!")
        return None
    try:
        # We assume the generator has a method that returns the raw segmentation map
        # where each pixel's value corresponds to a label in `label_map`.
        # Let's call this hypothetical method `get_parsing_map`.
        # If your `HumanParsingMaskGenerator` works differently, you'll need to adjust this call.
        model_parse = mask_generator.get_parsing_map(src_image_pil)
        return model_parse
    except Exception as e:
        print(f"Error generating parse map: {e}")
        # Fallback if `get_parsing_map` doesn't exist.
        gr.Warning("Could not generate a detailed parse map for face preservation.")
        return None




# --- Model Initialization ---
def initialize_models():
    global pipeline, ip_adapter_loaded, mask_generator
    print("--- Initializing Models (This may take a moment) ---")

    # 1. Load Checkpoint using from_single_file for .safetensors
    print(f"Loading Stable Diffusion Inpaint Pipeline from: {CHECKPOINT_PATH}")
    try:
        pipeline = StableDiffusionInpaintPipeline.from_single_file(
            CHECKPOINT_PATH,
            torch_dtype=torch.float16,
            safety_checker=None,
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
    try:
        # Load the IP-Adapter. `load_ip_adapter` can take a local directory.
        ip_adapter_model_id = "h94/IP-Adapter"
        pipeline.load_ip_adapter(ip_adapter_model_id, subfolder="models", weight_name="ip-adapter_sd15.bin")
        ip_adapter_loaded = True # Set flag to True on successful load

    except Exception as e:
        print(f"Error loading IP-Adapter: {e}")
        print("Ensure the IP-Adapter model file and CLIP Vision model path are correct and compatible.")
        ip_adapter_loaded = False # Set flag to False on failure
        return False # Indicate failure

    # 3. Initialize Mask Generator
    try:
        # Assuming HumanParsingMaskGenerator is the class from mask.py
        mask_generator = HumanParsingMaskGenerator(ckpt_dir="./ckpts", device="cuda" if torch.cuda.is_available() else "cpu")
        print("HumanParsingMaskGenerator initialized successfully.")
    except Exception as e:
        print(f"Error initializing HumanParsingMaskGenerator: {e}")
        return False # Indicate failure

    print("--- Model Initialization Complete ---")
    return True # Indicate success

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
    mask_smooth_amount: int,
    mask_expand_size: int,
    mask_boundary_scale: float # New parameter for mask generation
):
    """
    Main function to run the inpainting workflow, adapted for Gradio.
    """
    if pipeline is None or not ip_adapter_loaded or mask_generator is None:
        gr.Warning("Models not initialized or IP-Adapter/Mask Generator failed to load. Attempting re-initialization...")
        if not initialize_models(): # Attempt to re-initialize if not loaded
            return None # Return None if initialization fails

    if original_image_pil is None:
        gr.Warning("Please upload an original image.")
        return None
    if ip_adapter_image_pil is None:
        gr.Warning("Please upload an IP-Adapter reference image.")
        return None

    # Ensure images are in RGB mode
    original_image_pil = original_image_pil.convert("RGB")
    ip_adapter_image_pil = ip_adapter_image_pil.convert("RGB")

    # STEP 1: Generate the full parse map for face preservation later
    print("Generating full parse map for face preservation...")
    model_parse = mask_generator.human_parser(original_image_pil)
    
    # STEP 2: Generate Mask for inpainting the body
    print("Generating mask from original image...")
    generated_mask_pil = generate_mask_from_image(
        original_image_pil,
        resize_width, # Pass target width
        resize_height, # Pass target height
        parts_to_mask=[label_map["left_leg"], label_map["right_leg"], label_map["left_arm"], label_map["right_arm"], label_map["torso_skin"]], # Torso, arms, legs, skin
        boundary_scale=mask_boundary_scale
    )
    if generated_mask_pil is None:
        gr.Warning("Failed to generate mask. Please check mask generator setup.")
        return None

    # Process Generated Mask
    processed_mask = load_and_process_mask(generated_mask_pil, mask_smooth_amount, mask_expand_size)

    # Resize original image and processed mask to the target dimensions for inpainting
    original_image_resized = original_image_pil.resize((768, 1024))
    processed_mask_resized = processed_mask.resize((resize_width, resize_height), Image.NEAREST)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    # STEP 3: Run the inpainting pipeline
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
    inpainted_image_resized = inpainted_image.resize((768, 1024))
    inpainted_image_resized.save("inpainted_image_resized.png")
    # STEP 4: Preserve the original face and hair
    print("Preserving original face and hair...")
    final_image = preserve_face_and_hair(original_image_resized, inpainted_image_resized, model_parse)

    return final_image

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
        The inpainting mask will be **automatically generated** based on human parsing, and the **original face will be preserved**.
        """
    )

    with gr.Row():
        with gr.Column():
            original_image_input = gr.Image(type="pil", label="1. Original Image (to be inpainted)", value="Screenshot 2025-06-06 051538.png")
            ip_adapter_image_input = gr.Image(type="pil", label="2. IP-Adapter Reference Image", value="image1 (3).png")

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
            mask_boundary_scale_input = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.05, label="Mask Boundary Scale (for generation)")


            generate_button = gr.Button("Generate Inpainted Image", variant="primary")

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
            mask_smooth_amount_input,
            mask_expand_size_input,
            mask_boundary_scale_input # New input for mask generation
        ],
        outputs=output_image
    )

# Launch the Gradio app
if __name__ == "__main__":
    # Ensure you have the necessary libraries installed:
    # pip install torch diffusers transformers accelerate safetensors Pillow numpy gradio opencv-python
    # For HumanParsingMaskGenerator, you might need to install additional dependencies
    # and download checkpoints as specified by its repository.
    # Make sure './ckpts' directory exists and contains the necessary models for HumanParsingMaskGenerator.
    demo.launch()
# This environment variable can help prevent issues with model downloads on certain systems.
import os
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from controlnet_aux import OpenposeDetector
import sys
from typing import Union
import random

# --- Main Class for the Virtual Try-On Pipeline ---

class StableDiffusionImage:
    """
    A self-contained class that encapsulates the entire virtual try-on process,
    combining ControlNet for pose and IP-Adapter for style.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.pipe = None
        self.openpose_detector = None
        self._load_models()

    def _load_models(self):
        """Loads all required models onto the specified device."""
        print("Loading all models. This may take a moment...")
        print("Loading OpenPose Detector...")
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        print("Loading ControlNet model...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=torch.float16
        )
        print("Loading Stable Diffusion 1.5 pipeline...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        print("Loading IP-Adapter model...")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers memory efficient attention enabled.")
        except ImportError:
            print("xFormers is not installed. For faster inference, consider installing it with: pip install xformers")
        print("✅ All models loaded successfully!")

    def generate(
        self,
        pose_image: Union[str, Image.Image],
        style_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str,
        ip_adapter_scale: float = 0.7,
        guidance_scale: float = 7.5,
        num_steps: int = 40,
        seed: int = -1, # Default seed can be -1 for random
        controlnet_scale: float = 1.0 # The new parameter for pose strength
    ) -> Image.Image:
        """
        Generates a virtual try-on image.
        """
        print("\n--- Starting Generation Process ---")
        
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        print(f"Using Seed: {seed}")
        
        try:
            if isinstance(pose_image, str):
                source_image_pil = Image.open(pose_image).convert("RGB")
            else:
                source_image_pil = pose_image.convert("RGB")

            if isinstance(style_image, str):
                style_image_pil = Image.open(style_image).convert("RGB")
            else:
                style_image_pil = style_image.convert("RGB")
        except Exception as e:
            print(f"Error loading images: {e}", file=sys.stderr)
            return None

        print(f"Detecting pose from the source image...")
        pose_skeleton_image = self.openpose_detector(source_image_pil)

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Generating try-on image...")
        output_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_skeleton_image,
            ip_adapter_image=style_image_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            controlnet_conditioning_scale=controlnet_scale,
        ).images[0]

        print("Generation complete.")
        return output_image


# --- BATCH PROCESSING SCRIPT ---

def run_batch_process():
    """
    This function handles the batch processing of images
    with the specific naming convention 'pose (1).png', 'style (1).png', etc.
    """
    # --- 1. CONFIGURE YOUR BATCH JOB HERE ---
    
    POSE_FOLDER = r"A:\Vton\Mixvton\pose"
    STYLE_FOLDER = r"A:\Vton\Mixvton\style"
    OUTPUT_FOLDER = "mask"

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    PROMPT = "a photo of a woman in a dress, 8k, high quality, photorealistic"
    NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality, cartoon, anime"
    
    IP_ADAPTER_SCALE = 0.8
    GUIDANCE_SCALE = 7.5
    NUM_STEPS = 30
    
    # Set the strength of the OpenPose skeleton.
    # 1.0 = very strict, 0.5 = balanced suggestion, 0.0 = off
    CONTROLNET_STRENGTH = 1.0
    
    # --- 2. SCRIPT EXECUTION ---
    
    print("--- Starting Batch Generation ---")
    sd_pipeline = StableDiffusionImage()

    # Total number of images to process
    total_images = 24

    # Loop from 1 to 24 (inclusive) to match file names like 'pose (1).png'
    for i in range(1, total_images + 1):
        
        # Define the exact filenames for each iteration
        # IMPORTANT: Change '.png' if your files are '.jpg' or another format.
        pose_filename = f"pose ({i}).png"
        style_filename = f"style ({i}).png"
        output_filename = f"mask ({i}).png"

        # Construct the full file paths
        pose_path = os.path.join(POSE_FOLDER, pose_filename)
        style_path = os.path.join(STYLE_FOLDER, style_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        print(f"\n--- Processing Set {i}/{total_images} ---")
        
        # Check if both input files exist before trying to open them
        if not os.path.exists(pose_path):
            print(f"⚠️  SKIPPING: Pose image not found at {pose_path}")
            continue # Skip to the next iteration
        if not os.path.exists(style_path):
            print(f"⚠️  SKIPPING: Style image not found at {style_path}")
            continue # Skip to the next iteration
            
        print(f"Pose Image:  {pose_path}")
        print(f"Style Image: {style_path}")

        # Generate the new image
        result_image = sd_pipeline.generate(
            pose_image=pose_path,
            style_image=style_path,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_scale=IP_ADAPTER_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            num_steps=NUM_STEPS,
            seed=-1,  # Use -1 to generate a random seed for each image
            controlnet_scale=CONTROLNET_STRENGTH
        )
        
        if result_image:
            result_image.save(output_path)
            print(f"✅ Successfully saved result to: {output_path}")

    print("\n--- Batch processing finished! ---")


# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    run_batch_process()
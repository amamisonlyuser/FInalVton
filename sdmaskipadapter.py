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
import sys
# Import 'Union' for flexible type hinting
from typing import Union

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

        # 1. Load the OpenPose Detector for creating pose skeletons
        print("Loading OpenPose Detector...")
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        # 2. Load ControlNet for OpenPose
        print("Loading ControlNet model...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=torch.float16
        )

        # 3. Load the base Stable Diffusion 1.5 pipeline with ControlNet
        print("Loading Stable Diffusion 1.5 pipeline...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )

        # 4. Load and configure the IP-Adapter for style transfer
        print("Loading IP-Adapter model...")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        # 5. Configure the pipeline for performance and quality
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers memory efficient attention enabled.")
        except ImportError:
            print("xFormers is not installed. For faster inference, consider installing it with: pip install xformers")

        print("✅ All models loaded successfully!")

    # --- THIS IS THE FULLY UPDATED AND CORRECTED METHOD ---
    def generate(
        self,
        # The parameters are updated to accept either a path (str) or a PIL.Image object
        pose_image: Union[str, Image.Image],
        style_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str,
        ip_adapter_scale: float = 0.7,
        guidance_scale: float = 7.5,
        num_steps: int = 40,
        seed: int = 42
    ) -> Image.Image:
        """
        Generates a virtual try-on image.
        This version is updated to accept either file paths (str) or PIL.Image objects as input.
        """
        print("\n--- Starting Generation Process ---")
        
        # This logic block checks the type of input and handles it correctly
        try:
            # Handle the pose image
            if isinstance(pose_image, str):
                # If it's a string, it's a file path, so we open it
                source_image_pil = Image.open(pose_image).convert("RGB")
            else:
                # Otherwise, we assume it's already a PIL.Image object
                source_image_pil = pose_image.convert("RGB")

            # Handle the style image with the same logic
            if isinstance(style_image, str):
                style_image_pil = Image.open(style_image).convert("RGB")
            else:
                style_image_pil = style_image.convert("RGB")

        except FileNotFoundError as e:
            print(f"Error: Could not find an input image. {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error loading images: {e}", file=sys.stderr)
            return None

        # The rest of the function uses the correctly loaded PIL objects
        
        # Detect pose from the source image
        print(f"Detecting pose from the source image...")
        pose_skeleton_image = self.openpose_detector(source_image_pil)

        # Configure pipeline parameters for this run
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run the full pipeline
        print("Generating try-on image...")
        output_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_skeleton_image,          # ControlNet input
            ip_adapter_image=style_image_pil,   # IP-Adapter input
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        print("Generation complete.")
        return output_image
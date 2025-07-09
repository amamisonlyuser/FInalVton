import os
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL
)

# --- Custom imports ---
from leffa_utils.utils import resize_and_center, get_flexible_agnostic_mask , get_skin_mask
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from mask import HumanParsingMaskGenerator

mask_generator = HumanParsingMaskGenerator(ckpt_dir="./ckpts", device="cpu")

class SDXLBodyPartInpainter:
    def __init__(self, ckpt_dir="./ckpts", device="cuda"):
        self.device = device
        self.ckpt_dir = os.path.abspath(ckpt_dir)

        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        self.parsing = Parsing(
            atr_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_atr.onnx"),
            lip_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_lip.onnx"),
        )
        self.openpose = OpenPose(
            body_model_path=os.path.join(self.ckpt_dir, "openpose", "body_pose_model.pth")
        )

        print("Loading high-quality VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        ).to(self.device)

        print("Loading Base SDXL Inpainting Pipeline...")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=self.vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)

        # --- IP-Adapter ---
        print("Loading IP-Adapter for SDXL...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin"
        )
        print("IP-Adapter loaded.")

        print("Loading SDXL Refiner Pipeline...")
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)
        print("Refiner pipeline loaded.")
 
    def generate_mask(self, src_image_pil, parts_to_mask=None, boundary_scale=1.0):
        src_image = resize_and_center(src_image_pil, 768, 1024)
        generated_mask_pil = mask_generator.generate_mask(
        src_image_pil=src_image,
        target_labels=[12, 13, 14, 15, 18], # As requested: torso, arms, skin
        fill_holes=True
    )
        return generated_mask_pil



    def inpaint_with_reference(self, src_image_pil, ref_image_pil, parts_to_mask):
        print("[🛠️] Generating mask...")
        mask = self.generate_mask(src_image_pil, parts_to_mask)
        mask.save("body_mask.png")

        print("[🎨] Preparing images for inpainting...")
        src_resized = resize_and_center(src_image_pil, 1024, 1024)
        mask_resized = mask.resize((1024, 1024), Image.NEAREST)
        ref_resized = ref_image_pil.resize((512, 512))  # Recommended for IP-Adapter
        ip_adapter_scale = 0.7
        print("[🧠] Setting reference image for IP-Adapter...")
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        prompt = "a person matching the reference appearance in terms of skin tone and clothing"
        negative_prompt = "blurry, distorted, deformed, mismatch"

        print("[🖌️] Running SDXL Inpainting...")
        result = self.pipe(
            image=src_resized,
            mask_image=mask_resized,
            ip_adapter_image=ref_resized,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=6.5,
            num_inference_steps=30,
            padding_mask_crop=12
        ).images[0]

        print("[🔧] Refining result...")
        result = self.refiner(
            image=result,
            prompt=prompt,
            guidance_scale=6.5,
            num_inference_steps=40
        ).images[0]

        result.save("inpainted_result.png")
        print("[✅] Inpainting completed and saved as 'inpainted_result.png'")
        return result
 
                                                                                          
# --- Example Usage ---
if __name__ == "__main__":
    src_path = r"A:\KNOT\newvton\Screenshot 2025-06-06 051538.png"
    ref_path = r"A:\KNOT\newvton\ref.png"

    source_img = Image.open(src_path).convert("RGB")
    reference_img = Image.open(ref_path).convert("RGB")

    inpainter = SDXLBodyPartInpainter(ckpt_dir="./ckpts", device="cuda")
    output = inpainter.inpaint_with_reference(
        src_image_pil=source_img,
        ref_image_pil=reference_img,
        parts_to_mask=["left_arm", "right_arm", "arms", "neck"]
    )

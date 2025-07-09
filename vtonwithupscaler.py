# app.py
#
# Integrated Leffa Virtual Try-On pipeline with GFPGAN-based Face Restoration
# and post-process boundary smoothing.
# All models are loaded once at startup for efficient inference.

import gradio as gr
import numpy as np
from PIL import Image
import os
import sys
import cv2

# --- Main Imports for VTON and Face Enhancement ---

# Import the FaceRestorer class from your module.
# This assumes 'face_restorer.py' is in the same directory as this app.py.
try:
    from GFPGAN.infrenceforvton import FaceRestorer
except ImportError:
    print("Error: 'face_restorer.py' not found. Please ensure the file is in the correct directory.")
    sys.exit(1)

# Import Leffa components
try:
    from leffa.transform import LeffaTransform
    from leffa.model import LeffaModel
    from leffa.inference import LeffaInference
    from leffa_utils.densepose_predictor import DensePosePredictor
    from leffa_utils.utils import resize_and_center, get_flexible_agnostic_mask , preserve_face_and_hair
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
except ImportError as e:
    print(f"Warning: A Leffa-related module failed to import: {e}.")
    print("The application will define dummy classes, but will not be functional.")
    # Define dummy classes to allow the script to be parsed
    class LeffaTransform: pass
    class LeffaModel: pass
    class LeffaInference: pass
    class DensePosePredictor: pass
    class Parsing: pass
    class OpenPose: pass


# --- Label Map for Parsing ---
label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
    "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
    "bag": 16, "scarf": 17, "neck": 18,
}

# --- Core Predictor Class ---

class LeffaPredictor:
    """
    A unified predictor class that handles VTON generation, face enhancement,
    and boundary smoothing.
    """
    def __init__(self, ckpt_dir="./ckpts", device='cpu'):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.device = device
        print(f"Initializing LeffaPredictor with checkpoint directory: {self.ckpt_dir}")

        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        # --- Initialize VTON components ---
        print("Loading VTON models...")
        densepose_config_path = os.path.join(self.ckpt_dir, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml")
        self.densepose_predictor = DensePosePredictor(
            config_path=densepose_config_path,
            weights_path=os.path.join(self.ckpt_dir, "densepose", "model_final_162be9.pkl"),
        )
        self.parsing = Parsing(
            atr_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_atr.onnx"),
            lip_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_lip.onnx"),
        )
        self.openpose = OpenPose(
            body_model_path=os.path.join(self.ckpt_dir, "openpose", "body_pose_model.pth"),
        )
        sd_inpainting_path = os.path.join(self.ckpt_dir, "stable-diffusion-inpainting")
        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path=sd_inpainting_path,
            pretrained_model=os.path.join(self.ckpt_dir, "virtual_tryon.pth"),
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)
        print("VTON models loaded successfully.")

        # --- Initialize Face Restorer ---
        print("\nInitializing Face Restorer...")
        self.face_restorer = FaceRestorer(upscale=2, bg_upsampler='realesrgan')
        print("Face Restorer initialized successfully.")

    def run_vton(
        self,
        src_image_pil,
        ref_image_pil,
        parts_to_mask: list,
        boundary_scale=5,
        step=30,
        scale=2.5,
        seed=42
    ):
        """ Runs the core VTON inference. """
        ref_image = ref_image_pil
        src_image = resize_and_center(src_image_pil, 768, 1024)
        src_image_array = np.array(src_image)
        src_image_rgb = src_image.convert("RGB")

        model_parse, _ = self.parsing(src_image_rgb.resize((384, 512)))
        keypoints = self.openpose(src_image_rgb.resize((384, 512)))
        src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        densepose = Image.fromarray(src_image_seg_array)
        densepose.save("DENSEPOSE.png")
        mask = get_flexible_agnostic_mask(
            model_parse=model_parse,
            keypoint=keypoints,
            parts_to_mask=parts_to_mask,
            size=(768, 1024),
            boundary_scale=boundary_scale
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        transform = LeffaTransform()
        data = {
            "src_image": [src_image], "ref_image": [ref_image],
            "mask": [mask], "densepose": [densepose],
        }
        data = transform(data)

        output = self.vt_inference_hd(
            data, num_inference_steps=step, guidance_scale=scale, seed=seed
        )
        gen_image_pil = output["generated_image"][0]
        return gen_image_pil

    def smooth_boundaries(self, image_np, blur_kernel_size=15, dilation_size=5):
        """
        Smooths the boundaries between different regions of an image using human parsing.
        
        Args:
            image_np (np.ndarray): The input image in NumPy array format (BGR).
            blur_kernel_size (int): The kernel size for Gaussian blur. Must be an odd number.
            dilation_size (int): The size of the kernel for dilating the boundary mask.
            
        Returns:
            np.ndarray: The image with smoothed boundaries.
        """
        # 1. Prepare image and run human parsing
        h, w, _ = image_np.shape
        image_pil_rgb = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        parsed_map, _ = self.parsing(image_pil_rgb.resize((384, 512)))
        parsed_map = cv2.resize(parsed_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Find boundaries in the parsing map using Canny edge detection
        edges = cv2.Canny(np.uint8(parsed_map), 1, 1)

        # 3. Create a dilated boundary mask to cover a wider area
        if dilation_size > 0:
            dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)
            boundary_mask = cv2.dilate(edges, dilation_kernel, iterations=1)
        else:
            boundary_mask = edges
        
        # 4. Create a blurred version of the entire image
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blurred_image = cv2.GaussianBlur(image_np, (blur_kernel_size, blur_kernel_size), 0)

        # 5. Combine the original and blurred images using the boundary mask
        smoothed_image = image_np.copy()
        smoothed_image[boundary_mask != 0] = blurred_image[boundary_mask != 0]

        return smoothed_image


# --- Gradio Interface Logic ---

def gradio_interface(person_image, garment_image, custom_parts, boundary_scale, steps, guidance_scale, seed, enhance_face_option, smooth_boundaries_option):
    """ Main function that connects the Gradio UI to the backend predictor. """
    try:
        person_pil_image = Image.fromarray(person_image)
        garment_pil_image = Image.fromarray(garment_image)

        # --- Step 1: Run the VTON generation ---
        final_image = predictor.run_vton(
            src_image_pil=person_pil_image, ref_image_pil=garment_pil_image,
            parts_to_mask=custom_parts, boundary_scale=int(boundary_scale),
            step=int(steps), scale=float(guidance_scale), seed=int(seed)
        )

        # --- Step 2: Conditionally apply face enhancement with fallback ---
        if enhance_face_option:
            try:
                print("Applying face enhancement...")
                generated_image_np = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
                enhanced_image_np = predictor.face_restorer.enhance_face(generated_image_np, weight=0.75)
                final_image = Image.fromarray(cv2.cvtColor(enhanced_image_np, cv2.COLOR_BGR2RGB))
                print("Face enhancement complete.")
            except Exception as e:
                print(f"⚠️ WARNING: Face enhancement failed: {e}", file=sys.stderr)
                # Fallback is the image from Step 1, which is already in final_image
        
        # --- Step 3: Conditionally apply boundary smoothing with fallback ---
        if smooth_boundaries_option:
            try:
                print("Applying boundary smoothing...")
                image_to_smooth_np = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
                smoothed_image_np = predictor.smooth_boundaries(image_to_smooth_np)
                final_image = Image.fromarray(cv2.cvtColor(smoothed_image_np, cv2.COLOR_BGR2RGB))
                print("Boundary smoothing complete.")
            except Exception as e:
                print(f"⚠️ WARNING: Boundary smoothing failed: {e}", file=sys.stderr)
                # Fallback is the image from the previous step, already in final_image

        return final_image

    except Exception as e:
        print(f"ERROR in Gradio Interface: {e}", file=sys.stderr)
        raise gr.Error(f"An error occurred: {e}")


# --- Application Entry Point ---

if __name__ == "__main__":
    AVAILABLE_PARTS = sorted(list(label_map.keys()) + ["arms", "hands", "shoes"])

    try:
        # --- Initialize all models once ---
        predictor = LeffaPredictor()
        print("\n✅ All models loaded. Gradio interface is starting.")

        # --- Build Gradio UI ---
        with gr.Blocks(css=".gradio-container {max-width: 1200px !important;}") as app:
            gr.Markdown("## Virtual Try-On with Face Enhancement & Smoothing")
            gr.Markdown("Upload a person and a garment image, then click Generate.")

            with gr.Row():
                with gr.Column(scale=1):
                    person_image_input = gr.Image(type="numpy", label="Person Image")
                    garment_image_input = gr.Image(type="numpy", label="Garment Image")
                    parts_to_mask_input = gr.CheckboxGroup(
                        choices=AVAILABLE_PARTS, value=["upper_clothes", "arms"],
                        label="Parts to Replace with Garment",
                        info="Select all body parts that should be covered by the new garment."
                    )
                with gr.Column(scale=1):
                    output_image = gr.Image(type="pil", label="Generated Try-On")

            run_button = gr.Button("Generate Try-On", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                enhance_face_input = gr.Checkbox(
                    value=True, label="Enhance Face Quality",
                    info="Uses GFPGAN to restore facial details in the final image."
                )
                smooth_boundaries_input = gr.Checkbox(
                    value=True, label="Smooth Clothing Boundaries",
                    info="Uses human parsing to find and blur the edges for a more natural blend."
                )
                boundary_scale_input = gr.Slider(-10, 10, value=6, step=1, label="Mask Boundary Scale")
                steps_input = gr.Slider(1, 100, value=12, step=1, label="Inference Steps")
                scale_input = gr.Slider(1.0, 10.0, value=2.5, step=0.1, label="Guidance Scale")
                seed_input = gr.Number(value=42, label="Seed")

            run_button.click(
                fn=gradio_interface,
                inputs=[
                    person_image_input, garment_image_input, parts_to_mask_input,
                    boundary_scale_input, steps_input, scale_input, seed_input,
                    enhance_face_input, smooth_boundaries_input
                ],
                outputs=output_image
            )

        app.launch(debug=True)

    except Exception as e:
        print(f"🚨 An unexpected error occurred during startup: {type(e).__name__} - {e}")
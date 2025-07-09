# This environment variable can help prevent issues with model downloads on certain systems.
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
import cv2
import numpy as np
from PIL import Image
import random
import gradio as gr
from leffa_utils.utils import resize_and_center ,preserve_face_and_hair

# --- Check for Dependencies and Import ---
# Assuming 'leffa' and 'leffa_utils' are custom modules.
# Ensure they are in Python's path or installed.
try:
    from leffa.transform import LeffaTransform
    from leffa.model import LeffaModel
    from leffa.inference import LeffaInference
    from leffa_utils.densepose_predictor import DensePosePredictor
    from leffa_utils.utils import resize_and_center, get_flexible_agnostic_mask
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
except ImportError as e:
    print("="*80)
    print("ERROR: Failed to import a required custom module (leffa, leffa_utils, etc.).")
    print(f"Details: {e}")
    print("Please ensure that the 'leffa' project directory is in your Python path.")
    print("="*80)
    exit()

def clean_and_expand_densepose_mask(densepose_image: Image.Image, expand_radius: int = 5):
    """
    Fill holes and slightly expand the densepose mask.

    Args:
        densepose_image (PIL.Image): Input densepose image (segmentation map, RGB or label map).
        expand_radius (int): Expansion size for dilation.

    Returns:
        PIL.Image: Cleaned binary mask
    """
    densepose_np = np.array(densepose_image)

    # If the densepose is RGB (3D), convert it to grayscale
    if densepose_np.ndim == 3:
        # Convert RGB to grayscale mask: body is any non-black pixel
        mask = (np.any(densepose_np != 0, axis=-1)).astype(np.uint8) * 255
    else:
        # Assume it's already a label mask
        mask = (densepose_np > 0).astype(np.uint8) * 255

    # Fill small holes
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Expand (dilate) body region
    kernel_dilate = np.ones((expand_radius, expand_radius), np.uint8)
    expanded = cv2.dilate(closed, kernel_dilate, iterations=1)

    return Image.fromarray(cv2.cvtColor(expanded.astype(np.uint8), cv2.COLOR_GRAY2RGB))

def refined_crop_and_pad(image: Image.Image, mask: Image.Image, padding: int = 32, min_crop_size: int = 256):
    """
    Refined crop function using improved mask processing.
    """
    mask_np = np.array(mask.convert("L"))

    # Step 1: Fill small holes and close gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # Step 2: Get bounding box
    coords = np.argwhere(dilated > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty; cannot determine bounding box.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Step 3: Apply padding and enforce crop size
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    crop_width = max((x_max - x_min) + 2 * padding, min_crop_size)
    crop_height = max((y_max - y_min) + 2 * padding, min_crop_size)

    # Align to multiples of padding (e.g., 32 for SD)
    crop_width = int(np.ceil(crop_width / padding) * padding)
    crop_height = int(np.ceil(crop_height / padding) * padding)

    left = max(center_x - crop_width // 2, 0)
    upper = max(center_y - crop_height // 2, 0)
    right = min(left + crop_width, image.width)
    lower = min(upper + crop_height, image.height)

    # Adjust left/top if right/lower exceeds bounds
    if right - left < crop_width:
        left = max(right - crop_width, 0)
    if lower - upper < crop_height:
        upper = max(lower - crop_height, 0)

    bbox = (left, upper, right, lower)
    return image.crop(bbox), bbox


def paste_back_to_original(original_image: Image.Image, generated_crop: Image.Image, bbox: tuple):
    result = original_image.copy()
    crop_width = bbox[2] - bbox[0]
    crop_height = bbox[3] - bbox[1]
    
    if generated_crop.size != (crop_width, crop_height):
        # Resize to fit the original crop box
        generated_crop = generated_crop.resize((crop_width, crop_height), Image.LANCZOS)

    result.paste(generated_crop, bbox)
    return result


class LeffaPredictor(object):
    """
    A predictor class to encapsulate the Leffa model and its components.
    It handles loading models and running the virtual try-on inference.
    """
    def __init__(self, ckpt_dir="./ckpts", device='cpu'):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.device = device
        print(f"Initializing LeffaPredictor with checkpoint directory: {self.ckpt_dir}")

        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        # --- Initialize other Leffa components ---
        densepose_config_path = os.path.join(self.ckpt_dir, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml")
        if not os.path.exists(densepose_config_path):
            raise FileNotFoundError(f"DensePose config file not found at {densepose_config_path}")

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
        if not os.path.isdir(sd_inpainting_path):
            raise FileNotFoundError(f"Stable Diffusion inpainting directory not found: {sd_inpainting_path}")

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path=sd_inpainting_path,
            pretrained_model=os.path.join(self.ckpt_dir, "virtual_tryon.pth"),
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)
    
    def _leffa_predict_internal(
     self,
     src_image_pil,
     ref_image_pil,
     pose_image_pil,
     parts_to_mask: list,
     boundary_scale=5,
     ref_acceleration=False,
     step=30,
     scale=2.5,
     seed=42,
     vt_repaint=False,
):
   
     ref_image = ref_image_pil
     src_image =  src_image_pil
     src_image = resize_and_center(src_image, 768, 1024)

     src_image_array = np.array(src_image)
     src_image_rgb = src_image.convert("RGB")

    # Parsing and keypoints for source
     src_image_model_parse, _ = self.parsing(src_image_rgb.resize((384, 512)))
     src_image_keypoints = self.openpose(src_image_rgb.resize((384, 512)))
     src_image_mask = get_flexible_agnostic_mask(
        model_parse=src_image_model_parse,
        keypoint=src_image_keypoints,
        parts_to_mask=parts_to_mask,
        size=(768, 1024),
        boundary_scale=boundary_scale
     )

    # Parsing and keypoints for pose
     pose_image_rgb = pose_image_pil.convert("RGB")
     pose_image = resize_and_center(pose_image_pil, 768, 1024) 
     pose_image_model_parse, _ = self.parsing(pose_image_rgb.resize((384, 512)))
     pose_image_keypoints = self.openpose(pose_image_rgb.resize((384, 512)))
     pose_image_mask = get_flexible_agnostic_mask(
        model_parse=pose_image_model_parse,
        keypoint=pose_image_keypoints,
        parts_to_mask=parts_to_mask,
        size=(768, 1024),
        boundary_scale=boundary_scale
    )
     pose_image_array = np.array(pose_image)
    # Combine masks from both source and pose
     pose_mask_np = np.array(pose_image_mask.convert("L")) > 0
     src_mask_np = np.array(src_image_mask.convert("L")) > 0
     combined_mask_np = np.logical_or(pose_mask_np, src_mask_np).astype(np.uint8) * 255
     combined_mask_pil = Image.fromarray(combined_mask_np, mode="L").convert("RGB")

    # Resize combined_mask to match source image resolution
     combined_mask_resized = combined_mask_pil.resize(src_image.size, resample=Image.NEAREST)


     # Predict densepose and resize to match source image
     src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
     src_image_densepose_raw = Image.fromarray(src_image_seg_array)
     pose_image_seg_array = self.densepose_predictor.predict_seg(pose_image_array)[:, :, ::-1]
     pose_image_densepose_raw = Image.fromarray(src_image_seg_array)
 

     src_image_densepose = np.array(src_image_densepose_raw)
     pose_image_densepose = np.array(pose_image_densepose_raw) 
     combined_densepose_np = cv2.addWeighted(pose_image_densepose, 0.5, src_image_densepose, 0.5, 0)
     combined_densepose_pil = Image.fromarray(combined_densepose_np).convert("RGB")

    # Resize combined_mask to match source image resolution
     combined_mask_resized = combined_mask_pil.resize(src_image.size, resample=Image.NEAREST)
     combined_densepose_pil.save("densepose_raw.png")
     densepose_resized = combined_densepose_pil.resize(src_image.size, resample=Image.NEAREST)
     

    # Prepare data and run inference
     transform = LeffaTransform()
     data = {
        "src_image": [src_image],
        "ref_image": [ref_image],
        "mask": [combined_mask_resized],
        "densepose": [densepose_resized],
     }
     data = transform(data)

     inference_engine = self.vt_inference_hd
     output = inference_engine(
         data,
        ref_acceleration=ref_acceleration,
        num_inference_steps=step,
        guidance_scale=scale,
        seed=seed,
        repaint=vt_repaint,
     )

     gen_image_pil = output["generated_image"][0]
     gen_image_pil.save("gen_image_pil.png")

    # Paste back into original resolution

     return gen_image_pil , src_image,src_image_model_parse 
    
    def leffa_predict_vt(self, src_image_pil, ref_image_pil, pose_image_pil, parts_to_mask, boundary_scale, **kwargs):
        """ Public prediction method with type checking. """
        if not isinstance(src_image_pil, Image.Image):
            raise TypeError("src_image_pil must be a PIL.Image.Image object.")
        if not isinstance(ref_image_pil, Image.Image):
            raise TypeError("ref_image_pil must be a PIL.Image.Image object.")
        if not isinstance(pose_image_pil, Image.Image):
            raise TypeError("pose_image_pil must be a PIL.Image.Image object.")

        return self._leffa_predict_internal(
            src_image_pil=src_image_pil,
            ref_image_pil=ref_image_pil,
            pose_image_pil=pose_image_pil,
            parts_to_mask=parts_to_mask,
            boundary_scale=boundary_scale,
            **kwargs
        )

# --- Global Predictor Initialization ---
try:
    print("Initializing LeffaPredictor. This may take a moment...")
    predictor = LeffaPredictor(ckpt_dir="./ckpts")
    print("✅ LeffaPredictor initialized successfully.")
except Exception as e:
    print(f"🚨 An unexpected error occurred during initialization: {type(e).__name__} - {e}")
    import traceback
    traceback.print_exc()
    predictor = None # Set predictor to None if initialization fails


# --- Gradio Inference Function ---
def run_leffa_gradio(
    person_image,
    garment_image,
    pose_image,
    parts_to_mask,
    boundary_scale,
    steps,
    guidance_scale,
    seed
):
    """ The main function called by the Gradio interface. """
    if predictor is None:
        raise gr.Error("Predictor failed to initialize. Please check the console for errors.")
    if person_image is None:
        raise gr.Error("Please upload a source person image.")
    if garment_image is None:
        raise gr.Error("Please upload a reference garment image.")
    if pose_image is None:
        raise gr.Error("Please upload a reference pose image.")
        
    print("🚀 Starting Leffa inference for Gradio...")
    
    # Convert numpy arrays from Gradio to PIL Images
    person_pil = Image.fromarray(person_image).convert("RGB")
    garment_pil = Image.fromarray(garment_image).convert("RGB")
    pose_pil = Image.fromarray(pose_image).convert("RGB")

    # Handle random seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
        
    print(f"  - Parameters: Steps={steps}, Guidance={guidance_scale}, Seed={seed}")

    # --- Run Prediction ---
    generated_array, src_image,src_image_model_parse  = predictor.leffa_predict_vt(
        src_image_pil=person_pil,
        ref_image_pil=garment_pil,
        pose_image_pil=pose_pil,
        parts_to_mask=parts_to_mask,
        boundary_scale=int(boundary_scale),
        step=int(steps),
        scale=float(guidance_scale),
        seed=int(seed)
    )

    print("✅ Inference complete.")
    
    # Convert numpy array outputs back to PIL Images for display
    final_image0 = preserve_face_and_hair(
           original_model_image = src_image,virtual_tryon_image=generated_array, model_parse = src_image_model_parse
            )
    

    return final_image0 


# --- Gradio Web Interface ---
with gr.Blocks(css="body {background-color: #f4f4f5;}") as app:
    gr.Markdown("# 👗 Leffa: Virtual Try-On with Pose Transfer")
    gr.Markdown("Generate a new image by combining a **source person**, a **garment style**, and a **target pose**.")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                person_input = gr.Image(type="numpy", label="Source Image (Person's Identity)")
                garment_input = gr.Image(type="numpy", label="Reference Image (Garment)")
                pose_input = gr.Image(type="numpy", label="Reference Image (Target Pose)")
            
            generate_button = gr.Button("Generate Try-On Image ✨", variant="primary", size="lg")
            
            with gr.Accordion("Advanced Settings", open=True):
                available_parts = ["upper_clothes", "lower_clothes", "dress", "skirt", "pants", "arms", "legs", "hair", "face", "background", "scarf", "sunglasses", "neck" ,"left_leg", "right_leg","right_arm" ,"left_arm", "hands"]
                parts_input = gr.CheckboxGroup(
                    choices=available_parts,
                    value=["upper_clothes", "skirt", "pants", "dress", "arms"],
                    label="Parts to Mask (parts from the source to be replaced)"
                ) 
                boundary_slider = gr.Slider(1, 10, value=4, step=1, label="Mask Boundary Scale")
                steps_slider = gr.Slider(10, 50, value=12, step=1, label="Inference Steps")
                guidance_slider = gr.Slider(1.0, 10.0, value=2.5, step=0.1, label="Guidance Scale (CFG)")
                
                with gr.Row():
                    seed_input = gr.Number(label="Seed (-1 for random)", value=42, precision=0)
                    random_seed_button = gr.Button("🎲")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Result", type="pil")
            mask_output = gr.Image(label="Agnostic Mask (from Pose)", type="pil")
            densepose_output = gr.Image(label="DensePose Map (from Pose)", type="pil")
            seed_output = gr.Number(label="Used Seed", interactive=False)

    # --- UI Event Handlers ---
    inputs = [
        person_input, garment_input, pose_input, parts_input, boundary_slider,
        steps_slider, guidance_slider, seed_input
    ]
    outputs = [output_image]
    
    generate_button.click(fn=run_leffa_gradio, inputs=inputs, outputs=outputs)
    random_seed_button.click(lambda: -1, inputs=[], outputs=seed_input)

if __name__ == "__main__":
    app.launch(debug=True)

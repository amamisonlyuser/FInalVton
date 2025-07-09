import gradio as gr
import numpy as np
from PIL import Image
import os
import cv2
from mask import HumanParsingMaskGenerator

# Assuming 'leffa' and 'leffa_utils' are custom modules.
try:
    from leffa.transform import LeffaTransform
    from leffa.model import LeffaModel
    from leffa.inference import LeffaInference
    from leffa_utils.densepose_predictor import DensePosePredictor
    from leffa_utils.utils import resize_and_center
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from GFPGAN.infrenceforvton import FaceRestorer
except ImportError as e:
    print(f"ImportError: {e}. Please ensure the required modules are in your Python path.")
    # Add dummy classes to allow the script to be parsed, but it won't run.
    class LeffaTransform: pass
    class LeffaModel: pass
    class LeffaInference: pass
    class DensePosePredictor: pass
    class Parsing: pass
    class OpenPose: pass
    class FaceRestorer: pass

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

# --- Label Map (used by multiple functions) ---
label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
    "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
    "bag": 16, "scarf": 17, "neck": 18,
}

# --- Dictionary to map simple categories to detailed part lists ---
CATEGORY_TO_PARTS_MAP = {
    "Upper Body": ["upper_clothes", "arms", "neck", "scarf"],
    "Lower Body": ["pants", "skirt", "legs"],
    "Dress / Overall": ["dress", "upper_clothes", "pants", "skirt", "legs", "arms", "neck"]
}

# --- Face Preservation Logic (remains unchanged) ---
def preserve_face_and_hair(original_model_image, virtual_tryon_image, model_parse):
    original_model_image = original_model_image.convert("RGBA")
    virtual_tryon_image = virtual_tryon_image.convert("RGBA")
    parse_map_resized = model_parse.resize(original_model_image.size, Image.NEAREST)
    parse_array = np.array(parse_map_resized)
    head_part_labels = [label_map["hat"], label_map["hair"], label_map["sunglasses"], label_map["head"]]
    head_mask = np.zeros(parse_array.shape, dtype=np.uint8)
    for label in head_part_labels:
        head_mask[parse_array == label] = 255
    clothing_part_labels = [label_map["neck"], label_map["upper_clothes"], label_map["dress"], label_map["scarf"], label_map["belt"]]
    clothing_mask = np.zeros(parse_array.shape, dtype=np.uint8)
    for label in clothing_part_labels:
        clothing_mask[parse_array == label] = 255
    clothing_mask_dilated = cv2.dilate(clothing_mask, np.ones((5, 5), np.uint8), iterations=5)
    final_head_mask = cv2.bitwise_and(head_mask, cv2.bitwise_not(clothing_mask_dilated))
    feathered_mask = cv2.GaussianBlur(final_head_mask, (15, 15), 0)
    blend_mask = Image.fromarray(feathered_mask)
    final_image = Image.composite(original_model_image, virtual_tryon_image, blend_mask)
    return final_image.convert("RGB")


mask_generator = HumanParsingMaskGenerator(ckpt_dir="./ckpts", device="cpu")

class LeffaPredictor(object):
    def __init__(self, ckpt_dir="./ckpts", device='cpu'):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.device = device
        print(f"Initializing LeffaPredictor with checkpoint directory: {self.ckpt_dir}")
        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")
        densepose_config_path = os.path.join(self.ckpt_dir, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml")
        if not os.path.exists(densepose_config_path):
            raise FileNotFoundError(f"DensePose config file not found at {densepose_config_path}")
        self.densepose_predictor = DensePosePredictor(config_path=densepose_config_path, weights_path=os.path.join(self.ckpt_dir, "densepose", "model_final_162be9.pkl"))
        self.parsing = Parsing(atr_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_atr.onnx"), lip_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_lip.onnx"))
        self.openpose = OpenPose(body_model_path=os.path.join(self.ckpt_dir, "openpose", "body_pose_model.pth"))
        sd_inpainting_path = os.path.join(self.ckpt_dir, "stable-diffusion-inpainting")
        if not os.path.isdir(sd_inpainting_path):
            raise FileNotFoundError(f"Stable Diffusion inpainting directory not found: {sd_inpainting_path}")
        vt_model_hd = LeffaModel(pretrained_model_name_or_path=sd_inpainting_path, pretrained_model=os.path.join(self.ckpt_dir, "virtual_tryon.pth"), dtype="float16")
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)
        self.face_restorer = FaceRestorer(upscale=2, bg_upsampler='realesrgan')

    def generate_mask(self, src_image_pil):
        src_image = resize_and_center(src_image_pil, 768, 1024)
        generated_mask_pil = mask_generator.generate_mask(
            src_image_pil=src_image,
            target_labels=[12, 13, 14, 15, 18], # As requested: torso, arms, skin
        )
        return generated_mask_pil

    # --- FIX WAS APPLIED TO THIS FUNCTION ---
    def _leffa_predict_internal(
        self,
        src_image_pil,
        ref_image_pil,
        ref_acceleration=False,
        step=30,
        scale=2.5,
        seed=42,
        vt_repaint=False,
    ):
        ref_image = ref_image_pil
        original_src_image = src_image_pil.copy() # Keep original for later use

        src_image = resize_and_center(original_src_image, 768, 1024)
        src_image_array = np.array(src_image)

        # --- FIX (1/2): Generate the full parse map needed for face preservation ---
        model_parse_map_pil, _ = self.parsing(src_image.resize((384, 512)))

        # Generate mask for inpainting

        src_image_mask = self.generate_mask(src_image)
        

        # Predict densepose and resize
        src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        densepose_raw = Image.fromarray(src_image_seg_array)

        # Prepare data for the model
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [src_image_mask],
            "densepose": [densepose_raw],
        }
        data = transform(data)

        # Run inference
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

        # Paste the generated crop back onto the original image
        return gen_image_pil

    def leffa_predict_vt(self, src_image_pil, ref_image_pil, **kwargs):
        if not isinstance(src_image_pil, Image.Image):
            raise TypeError("src_image_pil must be a PIL.Image.Image object.")
        if not isinstance(ref_image_pil, Image.Image):
            raise TypeError("ref_image_pil must be a PIL.Image.Image object.")
        return self._leffa_predict_internal(
            src_image_pil=src_image_pil,
            ref_image_pil=ref_image_pil,
            **kwargs
        )

    def smooth_boundaries(self, image_np, blur_kernel_size=15, dilation_size=5):
        h, w, _ = image_np.shape
        image_pil_rgb = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        parsed_map_pil, _ = self.parsing(image_pil_rgb.resize((384, 512)))
        parsed_map_np = np.array(parsed_map_pil)
        parsed_map = cv2.resize(parsed_map_np, (w, h), interpolation=cv2.INTER_NEAREST)
        edges = cv2.Canny(np.uint8(parsed_map), 1, 1)
        if dilation_size > 0:
            dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)
            boundary_mask = cv2.dilate(edges, dilation_kernel, iterations=1)
        else:
            boundary_mask = edges
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blurred_image = cv2.GaussianBlur(image_np, (blur_kernel_size, blur_kernel_size), 0)
        smoothed_image = image_np.copy()
        smoothed_image[boundary_mask != 0] = blurred_image[boundary_mask != 0]
        return smoothed_image


def gradio_interface(person_image, garment_image, garment_category, steps, guidance_scale, seed, preserve_face):
    """ Main function that connects the Gradio UI to the backend predictor. """
    try:
        person_pil_image = Image.fromarray(person_image)
        garment_pil_image = Image.fromarray(garment_image)

        # This call will now receive the 5 values it expects
        generated_image_pil = predictor.leffa_predict_vt(
            src_image_pil=person_pil_image,
            ref_image_pil=garment_pil_image,
            step=int(steps),
            scale=float(guidance_scale),
            seed=int(seed)
        )

        if preserve_face:
            final_image0 = preserve_face_and_hair(

                virtual_tryon_image=generated_image_pil,
               
            )
            generated_image_np = cv2.cvtColor(np.array(final_image0), cv2.COLOR_RGB2BGR)
            enhanced_image_np = predictor.face_restorer.enhance_face(generated_image_np, weight=0.75)
            final_image1 = Image.fromarray(cv2.cvtColor(enhanced_image_np, cv2.COLOR_BGR2RGB))
            image_to_smooth_np = cv2.cvtColor(np.array(final_image1), cv2.COLOR_RGB2BGR)
            smoothed_image_np = predictor.smooth_boundaries(image_to_smooth_np)
            final_image = Image.fromarray(cv2.cvtColor(smoothed_image_np, cv2.COLOR_BGR2RGB))
            final_image.save("final_image.png")
            return final_image
        else:
            return generated_image_pil

    except Exception as e:
        print(f"ERROR in Gradio Interface: {e}")
        raise gr.Error(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        predictor = LeffaPredictor()
        print("LeffaPredictor initialized successfully.")

        with gr.Blocks(css=".gradio-container {max-width: 1200px !important;}") as app:
            gr.Markdown("## Leffa Virtual Try-On")
            gr.Markdown("Upload a person image and a garment image.")

            with gr.Row():
                with gr.Column(scale=1):
                    person_image_input = gr.Image(type="numpy", label="Person Image")
                    garment_image_input = gr.Image(type="numpy", label="Garment Image")

                    garment_category_input = gr.Dropdown(
                        choices=list(CATEGORY_TO_PARTS_MAP.keys()),
                        value="Upper Body",
                        label="Garment Category",
                        info="Select garment type (currently not used for mask generation in this script).",
                        interactive=True
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        preserve_face_input = gr.Checkbox(value=True, label="Preserve original face and hair")
                        steps_input = gr.Slider(1, 100, value=12, step=1, label="Inference Steps")
                        scale_input = gr.Slider(1.0, 10.0, value=2.5, step=0.1, label="Guidance Scale")
                        seed_input = gr.Number(value=42, label="Seed")

                with gr.Column(scale=1):
                    output_image = gr.Image(type="pil", label="Generated Try-On")

            run_button = gr.Button("Generate Try-On", variant="primary")

            run_button.click(
                fn=gradio_interface,
                inputs=[
                    person_image_input,
                    garment_image_input,
                    garment_category_input,
                    steps_input,
                    scale_input,
                    seed_input,
                    preserve_face_input
                ],
                outputs=output_image
            )

        app.launch()

    except Exception as e:
        print(f"🚨 An unexpected error occurred during startup: {type(e).__name__} - {e}")
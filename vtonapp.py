import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import os
import sys
import cv2

# Assuming 'leffa' and 'leffa_utils' are custom modules.
# Make sure they are in Python's path or installed.
# If these imports fail, you may need to adjust your PYTHONPATH
# e.g., sys.path.insert(0, '/path/to/your/project/root')
try:
    from leffa.transform import LeffaTransform
    from leffa.model import LeffaModel
    from leffa.inference import LeffaInference
    from leffa_utils.densepose_predictor import DensePosePredictor
    from leffa_utils.utils import resize_and_center, get_flexible_agnostic_mask ,preserve_face_and_hair
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
except ImportError as e:
    print(f"ImportError: {e}. Please ensure the required modules are in your Python path.")
    # Add dummy classes to allow the script to be parsed, but it won't run.
    class LeffaTransform: pass
    class LeffaModel: pass
    class LeffaInference: pass
    class DensePosePredictor: pass
    class Parsing: pass
    class OpenPose: pass


# --- Label Map (used by multiple functions) ---
label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
    "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
    "bag": 16, "scarf": 17, "neck": 18,
}


# --- NEW FUNCTION: Face Preservation Logic ---
def preserve_face_and_hair(
    original_model_image: Image.Image,
    virtual_tryon_image: Image.Image,
    model_parse: Image.Image
) -> Image.Image:
    """
    Extracts the face and hair from the original model image, carefully excluding
    the neck and clothing, and smoothly pastes them onto the virtual try-on image.

    Args:
        original_model_image (Image.Image): The original image of the model.
        virtual_tryon_image (Image.Image): The output image from the virtual try-on pipeline.
        model_parse (Image.Image): The semantic segmentation map of the original model image.

    Returns:
        Image.Image: The virtual try-on image with the original, high-fidelity face
                     and hair blended in.
    """
    original_model_image = original_model_image.convert("RGBA")
    virtual_tryon_image = virtual_tryon_image.convert("RGBA")

    parse_map_resized = model_parse.resize(original_model_image.size, Image.NEAREST)
    parse_array = np.array(parse_map_resized)

    # 1. --- Create a mask for the complete head area (face, hair, etc.) ---
    # We include all parts of the head for a complete shape.
    head_part_labels = [
        label_map["hat"], label_map["hair"], label_map["sunglasses"], label_map["head"]
    ]
    head_mask = np.zeros(parse_array.shape, dtype=np.uint8)
    for label in head_part_labels:
        head_mask[parse_array == label] = 255

    # 2. --- Create a "barrier" mask from the neck and clothing parts ---
    # This mask will be used to cut off the head mask cleanly at the jaw/neckline.
    # We include the 'neck' here to ensure it gets replaced by the new garment.
    clothing_part_labels = [
        label_map["neck"], label_map["upper_clothes"], label_map["dress"],
        label_map["scarf"], label_map["belt"]
    ]
    clothing_mask = np.zeros(parse_array.shape, dtype=np.uint8)
    for label in clothing_part_labels:
        clothing_mask[parse_array == label] = 255

    # 3. --- Refine the masks to create a clean separation ---
    # Dilate the clothing mask slightly to make it overlap the head's bottom edge.
    # This ensures a clean cut without leaving a gap or a halo of old pixels.
    clothing_mask_dilated = cv2.dilate(clothing_mask, np.ones((5, 5), np.uint8), iterations=5)

    # Subtract the clothing area from the head area.
    # This is the key step to prevent the mask from covering the neck/chest.
    final_head_mask = cv2.bitwise_and(head_mask, cv2.bitwise_not(clothing_mask_dilated))

    # 4. --- Feather the mask for smooth blending ---
    # A Gaussian blur creates soft edges, which is crucial for a natural-looking blend.
    feathered_mask = cv2.GaussianBlur(final_head_mask, (15, 15), 0)

    # Convert the feathered mask to a PIL Image to use as a blend alpha mask
    blend_mask = Image.fromarray(feathered_mask)

    # 5. --- Composite the Images ---
    final_image = Image.composite(original_model_image, virtual_tryon_image, blend_mask)

    return final_image.convert("RGB")


class LeffaPredictor(object):
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
        parts_to_mask: list,
        boundary_scale=5,
        ref_acceleration=False,
        step=30,
        scale=2.5,
        seed=42,
        vt_repaint=False,
    ):
        ref_image = ref_image_pil
        src_image = resize_and_center(src_image_pil, 768, 1024)
        src_image_array = np.array(src_image)
        src_image_rgb = src_image.convert("RGB")

        # --- MODIFICATION: model_parse is needed later for face preservation ---
        model_parse, _ = self.parsing(src_image_rgb.resize((384, 512)))
        
        keypoints = self.openpose(src_image_rgb.resize((384, 512)))
        src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        densepose = Image.fromarray(src_image_seg_array)
        densepose.save("denspose.png")
        mask = get_flexible_agnostic_mask(
            model_parse=model_parse,
            keypoint=keypoints,
            parts_to_mask=parts_to_mask,
            size=(768, 1024),
            boundary_scale=boundary_scale
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask.save("mask.png")

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
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

        # --- MODIFICATION: Return the original image and parse map ---
        # These are needed for the face preservation step later.
        return gen_image_pil, mask, densepose, src_image, model_parse


    def leffa_predict_vt(self, src_image_pil, ref_image_pil, parts_to_mask, boundary_scale, **kwargs):
        if not isinstance(src_image_pil, Image.Image):
            raise TypeError("src_image_pil must be a PIL.Image.Image object.")
        if not isinstance(ref_image_pil, Image.Image):
            raise TypeError("ref_image_pil must be a PIL.Image.Image object.")

        return self._leffa_predict_internal(
            src_image_pil=src_image_pil,
            ref_image_pil=ref_image_pil,
            parts_to_mask=parts_to_mask,
            boundary_scale=boundary_scale,
            **kwargs
        )


def gradio_interface(person_image, garment_image, custom_parts, boundary_scale, model_type, steps, guidance_scale, seed, preserve_face):
    """ Main function that connects the Gradio UI to the backend predictor. """
    try:
        person_pil_image = Image.fromarray(person_image)
        garment_pil_image = Image.fromarray(garment_image)

        # --- MODIFICATION: Unpack the new return values ---
        generated_image_pil, _, _, original_model_image, model_parse = predictor.leffa_predict_vt(
            src_image_pil=person_pil_image,
            ref_image_pil=garment_pil_image,
            parts_to_mask=custom_parts,
            boundary_scale=int(boundary_scale),
            step=int(steps),
            scale=float(guidance_scale),
            seed=int(seed)
        )

        # --- MODIFICATION: Conditionally apply face preservation ---
        if preserve_face:
            final_image = preserve_face_and_hair(
                original_model_image=original_model_image,
                virtual_tryon_image=generated_image_pil,
                model_parse=model_parse
            )
            final_image.save("final_image.png")
            return final_image
        else:
            return generated_image_pil

    except Exception as e:
        print(f"ERROR in Gradio Interface: {e}")
        raise gr.Error(f"An error occurred: {e}")


if __name__ == "__main__":
    AVAILABLE_PARTS = sorted(list(label_map.keys()) + ["arms", "hands", "shoes"])

    try:
        predictor = LeffaPredictor()
        print("LeffaPredictor initialized successfully.")

        with gr.Blocks(css=".gradio-container {max-width: 1200px !important;}") as app:
            gr.Markdown("## Leffa Virtual Try-On")
            gr.Markdown("Upload images, select parts to replace, adjust settings, and generate a virtual try-on.")

            with gr.Row():
                with gr.Column(scale=1):
                    person_image_input = gr.Image(type="numpy", label="Person Image")
                    garment_image_input = gr.Image(type="numpy", label="Garment Image")

                    parts_to_mask_input = gr.CheckboxGroup(
                        choices=AVAILABLE_PARTS,
                        value=["upper_clothes", "arms"],
                        label="Custom Parts to Replace with Garment",
                        info="Select all body parts that should be replaced by the garment."
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        # --- MODIFICATION: Added checkbox for face preservation ---
                        preserve_face_input = gr.Checkbox(
                            value=True,
                            label="Preserve original face and hair",
                            info="Blends the original face/hair onto the result for better quality."
                        )
                        boundary_scale_input = gr.Slider(-10, 10, value=6, step=1, label="Mask Boundary Scale", info="Positive values expand, negative values shrink.")
                        steps_input = gr.Slider(1, 100, value=12,step=1, label="Inference Steps")
                        scale_input = gr.Slider(1.0, 10.0, value=2.5, step=0.1, label="Guidance Scale")
                        seed_input = gr.Number(value=42, label="Seed")
                        model_type_input = gr.Radio(["viton_hd", "dress_code"], value="viton_hd", label="Model Type", visible=False) # Hidden for now as it's not used

                with gr.Column(scale=1):
                    output_image = gr.Image(type="pil", label="Generated Try-On")

            run_button = gr.Button("Generate Try-On", variant="primary")

            # --- MODIFICATION: Added preserve_face_input to the inputs list ---
            run_button.click(
                fn=gradio_interface,
                inputs=[
                    person_image_input,
                    garment_image_input,
                    parts_to_mask_input,
                    boundary_scale_input,
                    model_type_input,
                    steps_input,
                    scale_input,
                    seed_input,
                    preserve_face_input # New input
                ],
                outputs=output_image
            )

        app.launch()

    except Exception as e:
        print(f"🚨 An unexpected error occurred during startup: {type(e).__name__} - {e}")

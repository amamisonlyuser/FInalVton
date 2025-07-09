import numpy as np
from PIL import Image
import os
import sys

# Assuming 'leffa' and 'leffa_utils' are custom modules.
# Ensure they are in Python's path or installed.
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_flexible_agnostic_mask
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


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
        parts_to_mask: list,
        boundary_scale=5,
        ref_acceleration=False,
        step=30,
        scale=2.5,
        seed=42,
        vt_repaint=False,
    ):
        """ Internal prediction function. """
        ref_image = ref_image_pil
        src_image = resize_and_center(src_image_pil, 768, 1024)
        src_image_array = np.array(src_image)
        src_image_rgb = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image_rgb.resize((384, 512)))
        keypoints = self.openpose(src_image_rgb.resize((384, 512)))
        src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        densepose = Image.fromarray(src_image_seg_array)

        mask = get_flexible_agnostic_mask(
            model_parse=model_parse,
            keypoint=keypoints,
            parts_to_mask=parts_to_mask,
            size=(768, 1024),
            boundary_scale=boundary_scale
        )
        
        mask = mask.resize((768, 1024), Image.NEAREST)
        
        # --- Prepare Data and Run Inference ---
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

        return np.array(gen_image_pil), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_pil, ref_image_pil, parts_to_mask, boundary_scale, **kwargs):
        """ Public prediction method with type checking. """
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


if __name__ == "__main__":
    try:
        # --- 1. Initialize the predictor ---
        predictor = LeffaPredictor()
        print("✅ LeffaPredictor initialized successfully.")

        # --- 2. Define Batch Input/Output Paths ---
        # Note: Assumes source and reference images have a '.jpg' extension.
        # Change '.jpg' to '.png' or your actual file extension if different.
        source_folder = r"A:\Poster\frames"
        reference_folder = r"A:\Poster\ref"
        output_folder = r"A:\Poster\output"
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # --- 3. Set Parameters ---
        parts_to_mask = ["upper_clothes","skirt" , "pants" , "dress", "arms"]
        boundary_scale = 4
        inference_steps = 12
        guidance_scale = 2.5
        seed = 42

        # --- 4. Start Batch Inference ---
        print(f"🚀 Starting batch inference...")
        print(f"   Source Folder: {source_folder}")
        print(f"   Reference Folder: {reference_folder}")
        print(f"   Output Folder: {output_folder}")
        
        # Loop from 1 to 10 for frame_00001-frame_00010 and tryon (1)-tryon (10)
        for i in range(0, 58):
            # Construct file paths for the current pair
            # Using f-string with :05d to format the frame number with leading zeros
            person_image_path = os.path.join(source_folder, f"frame_{i:05d}.png")
            garment_image_path = os.path.join(reference_folder, f"tryon ({i}).png")
            output_image_path = os.path.join(output_folder, f"generated_frame_{i:05d}.png")
            
            # Check if both input files exist before processing
            if not os.path.exists(person_image_path):
                print(f"⚠️ WARNING: Source image not found, skipping: {person_image_path}")
                continue
            if not os.path.exists(garment_image_path):
                print(f"⚠️ WARNING: Reference image not found, skipping: {garment_image_path}")
                continue

            print(f"\nProcessing pair {i}/10...")
            print(f"  > Person: {os.path.basename(person_image_path)}")
            print(f"  > Garment: {os.path.basename(garment_image_path)}")

            # --- Load Images and ensure they are in RGB format---
            person_pil_image = Image.open(person_image_path).convert("RGB")
            garment_pil_image = Image.open(garment_image_path).convert("RGB")


            # --- Run Prediction ---
            generated_image_array, _, _ = predictor.leffa_predict_vt(
                src_image_pil=person_pil_image,
                ref_image_pil=garment_pil_image,
                parts_to_mask=parts_to_mask,
                boundary_scale=int(boundary_scale),
                step=int(inference_steps),
                scale=float(guidance_scale),
                seed=int(seed)
            )

            # --- Save the Output ---
            output_image = Image.fromarray(generated_image_array)
            output_image.save(output_image_path)
            print(f"  > ✅ Successfully generated and saved to: {output_image_path}")

        print("\n🎉 Batch processing complete.")

    except FileNotFoundError as e:
        print(f"🚨 ERROR: A required file or directory was not found.")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"🚨 An unexpected error occurred during execution: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
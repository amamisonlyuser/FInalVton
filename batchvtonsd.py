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
    # ... __init__ method is unchanged ...
    def __init__(self, ckpt_dir="./ckpts", device='cpu'):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.device = device
        print(f"Initializing LeffaPredictor with checkpoint directory: {self.ckpt_dir}")

        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

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


    # <-- CHANGED: The internal method now accepts a 'pose_image_pil'
    def _leffa_predict_internal(
        self,
        src_image_pil,      # The person image to be edited
        ref_image_pil,      # The garment image
        pose_image_pil,     # The image that defines the output pose
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
        # The source image is the one we will inpaint on.
        src_image = resize_and_center(src_image_pil, 768, 1024)

        # <-- CHANGED: All parsing, keypoint, and densepose detection is
        # now performed on the 'pose_image_pil' instead of 'src_image_pil'.
        pose_image_for_parsing = resize_and_center(pose_image_pil, 768, 1024)
        pose_image_array = np.array(pose_image_for_parsing)
        pose_image_rgb = pose_image_for_parsing.convert("RGB")
        model_parse, _ = self.parsing(pose_image_rgb.resize((384, 512)))
        keypoints = self.openpose(pose_image_rgb.resize((384, 512)))
        pose_image_seg_array = self.densepose_predictor.predict_seg(pose_image_array)[:, :, ::-1]
        densepose = Image.fromarray(pose_image_seg_array)

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
            # Note: 'src_image' is the original person image. 'densepose' is from the new pose image.
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

    # <-- CHANGED: Public method now accepts 'pose_image_pil'
    def leffa_predict_vt(self, src_image_pil, ref_image_pil, pose_image_pil, parts_to_mask, boundary_scale, **kwargs):
        """ Public prediction method with type checking. """
        if not isinstance(src_image_pil, Image.Image):
            raise TypeError("src_image_pil must be a PIL.Image.Image object.")
        if not isinstance(ref_image_pil, Image.Image):
            raise TypeError("ref_image_pil must be a PIL.Image.Image object.")
        # <-- NEW: Add type check for the new pose image
        if not isinstance(pose_image_pil, Image.Image):
            raise TypeError("pose_image_pil must be a PIL.Image.Image object.")

        return self._leffa_predict_internal(
            src_image_pil=src_image_pil,
            ref_image_pil=ref_image_pil,
            pose_image_pil=pose_image_pil, # <-- CHANGED: Pass the pose image down
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
        source_folder = r"A:\Vton\Mixvton\pose"
        reference_folder = r"A:\Vton\Mixvton\style"
        # <-- NEW: Define the third input folder for the pose images
        mask_folder = r"A:\Vton\Mixvton\mask"
        output_folder = "finalvton"
        
        os.makedirs(output_folder, exist_ok=True)

        # --- 3. Set Parameters ---
        parts_to_mask = ["upper_clothes", "skirt", "pants", "dress", "arms"]
        boundary_scale = 4
        inference_steps = 12
        guidance_scale = 2.5
        seed = 42

        # --- 4. Start Batch Inference ---
        print(f"🚀 Starting batch inference...")
        print(f"   Source Person Folder: {source_folder}")
        print(f"   Garment Folder:     {reference_folder}")
        print(f"   Pose/Mask Folder:   {mask_folder}")
        print(f"   Output Folder:      {output_folder}")
        
        # Loop over your desired range of frames
        for i in range(1, 25):
            # Construct file paths for the current set of three images
            person_image_path = os.path.join(source_folder, f"pose ({i}).png")
            garment_image_path = os.path.join(reference_folder, f"style ({i}).png")
            # <-- NEW: Define the path for the pose image.
            # IMPORTANT: This assumes files in your 'mask' folder are named like 'frame_00000.png',
            # matching the files in the 'source_folder'. Change if your naming is different.
            pose_image_path = os.path.join(mask_folder, f"mask ({i}).png")
            output_image_path = os.path.join(output_folder, f"finalvton  ({i}).png")
            
            # <-- CHANGED: Check if all three input files exist
            if not os.path.exists(person_image_path):
                print(f"⚠️ WARNING: Source image not found, skipping: {person_image_path}")
                continue
            if not os.path.exists(garment_image_path):
                print(f"⚠️ WARNING: Reference image not found, skipping: {garment_image_path}")
                continue
            if not os.path.exists(pose_image_path):
                print(f"⚠️ WARNING: Pose image not found, skipping: {pose_image_path}")
                continue

            print(f"\nProcessing set {i}...")
            print(f"   > Person:  {os.path.basename(person_image_path)}")
            print(f"   > Garment: {os.path.basename(garment_image_path)}")
            print(f"   > Pose:    {os.path.basename(pose_image_path)}")

            # --- Load Images and ensure they are in RGB format---
            person_pil_image = Image.open(person_image_path).convert("RGB")
            garment_pil_image = Image.open(garment_image_path).convert("RGB")
            pose_pil_image = Image.open(pose_image_path).convert("RGB")

            # <-- CHANGED: Call prediction with all three images
            generated_image_array, _, _ = predictor.leffa_predict_vt(
                src_image_pil=person_pil_image,
                ref_image_pil=garment_pil_image,
                pose_image_pil=pose_pil_image, # Pass the new pose image here
                parts_to_mask=parts_to_mask,
                boundary_scale=int(boundary_scale),
                step=int(inference_steps),
                scale=float(guidance_scale),
                seed=int(seed)
            )

            # --- Save the Output ---
            output_image = Image.fromarray(generated_image_array)
            output_image.save(output_image_path)
            print(f"   > ✅ Successfully generated and saved to: {output_image_path}")

        print("\n🎉 Batch processing complete.")

    except FileNotFoundError as e:
        print(f"🚨 ERROR: A required file or directory was not found.")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"🚨 An unexpected error occurred during execution: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
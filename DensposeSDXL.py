import os
import cv2
import numpy as np
import torch
from PIL import Image

# Ensure you have installed detectron2 and DensePose correctly.
# You can typically do this by following the official installation guides:
# 1. Install PyTorch from https://pytorch.org/
# 2. Install detectron2:
#    pip install 'git+https://github.com/facebookresearch/detectron2.git'
# 3. Clone and install DensePose from the detectron2 projects directory.
try:
    from densepose import add_densepose_config
    from densepose.vis.densepose_results import (
        DensePoseResultsFineSegmentationVisualizer as Visualizer,
    )
    from densepose.vis.extractor import DensePoseResultExtractor
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
except ImportError:
    print("="*80)
    print("ERROR: Failed to import Detectron2 or DensePose.")
    print("Please ensure that PyTorch, Detectron2, and DensePose are installed correctly.")
    print("For installation instructions, please visit:")
    print("- PyTorch: https://pytorch.org/")
    print("- Detectron2: https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md")
    print("="*80)
    # Exit gracefully if dependencies are missing
    exit()


class DensePosePredictor:
    """
    A wrapper class for running DensePose inference.
    """
    def __init__(self, config_path, weights_path):
        """
        Initializes the DensePose predictor.

        Args:
            config_path (str): Path to the DensePose YAML configuration file.
            weights_path (str): Path to the pre-trained DensePose model weights (.pkl file).
        """
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(config_path)
        
        # Specify the path to the model weights
        self.cfg.MODEL.WEIGHTS = weights_path
        
        # Set the device to CUDA if available, otherwise CPU
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set a score threshold for detections
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        # Initialize the predictor, extractor, and visualizer
        self.predictor = DefaultPredictor(self.cfg)
        self.extractor = DensePoseResultExtractor()
        self.visualizer = Visualizer()

    def predict(self, image_bgr):
        """
        Performs DensePose prediction on a single image.

        Args:
            image_bgr (np.ndarray): The input image in BGR format (as read by OpenCV).

        Returns:
            A tuple containing DensePose results and corresponding bounding boxes.
        """
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError(f"Input image must be a NumPy array, but got {type(image_bgr)}")
            
        with torch.no_grad():
            outputs = self.predictor(image_bgr)["instances"]
        
        # Extract the results into a specialized format
        return self.extractor(outputs)

    def predict_seg(self, image_bgr):
        """
        Generates a DensePose segmentation mask for the input image.

        Args:
            image_bgr (np.ndarray): The input image in BGR format.

        Returns:
            np.ndarray: An image (as a NumPy array) containing the DensePose segmentation visualization.
                        The output image is in BGR format.
        """
        # Get the raw DensePose predictions
        outputs = self.predict(image_bgr)
        
        # Create a black image with the same dimensions as the input to draw on
        image_seg = np.zeros(image_bgr.shape, dtype=np.uint8)
        
        # Draw the segmentation results onto the black image
        self.visualizer.visualize(image_seg, outputs)
        
        return image_seg


def main():
    """
    Main function to run the DensePose prediction.
    """
    # --- USER CONFIGURATION ---
    # IMPORTANT: Please update these paths to match your local setup.
    
    # 1. Path to the directory where you have stored the DensePose model files.
    #    You can download these from the DensePose Model Zoo.
    ckpt_dir = "./ckpts"  # e.g., "/path/to/your/models/densepose"
    
    # 2. Path to the source image you want to process.
    #    This should be an image containing one or more people.
    src_image_path = r"A:\Vton\Mixvton\leffa_utils\Cloth.png"  # e.g., "input_images/person_walking.jpg"
    
    # 3. Path where the output segmentation image will be saved.
    output_path = "densepose_segmentation.png" # e.g., "output_images/person_walking_densepose.png"

    # --- END OF USER CONFIGURATION ---

    # Construct the full paths to the configuration and weights files
    densepose_config_path = os.path.join(ckpt_dir, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml")
    densepose_weights_path = os.path.join(ckpt_dir, "densepose", "model_final_162be9.pkl")

    # --- PRE-RUN CHECKS ---
    # Verify that all necessary files and directories exist before proceeding.
    if not all(os.path.exists(p) for p in [densepose_config_path, densepose_weights_path, src_image_path]):
        print("Error: A required file or directory was not found.")
        print(f"Config path exists: {os.path.exists(densepose_config_path)} -> '{densepose_config_path}'")
        print(f"Weights path exists: {os.path.exists(densepose_weights_path)} -> '{densepose_weights_path}'")
        print(f"Source image exists: {os.path.exists(src_image_path)} -> '{src_image_path}'")
        print("\nPlease update the 'ckpt_dir' and 'src_image_path' variables in the script.")
        return

    try:
        # --- EXECUTION ---
        # 1. Initialize the DensePose predictor
        print("Initializing DensePose predictor...")
        predictor = DensePosePredictor(
            config_path=densepose_config_path,
            weights_path=densepose_weights_path,
        )
        print("Predictor initialized successfully.")

        # 2. Read the source image using OpenCV
        print(f"Reading source image from: {src_image_path}")
        # cv2.imread loads the image in Blue, Green, Red (BGR) channel order
        src_image_array = cv2.imread(src_image_path)

        # 3. Get the DensePose segmentation mask
        print("Predicting DensePose segmentation...")
        # The output from predict_seg is also a BGR image
        src_image_seg_array_bgr = predictor.predict_seg(src_image_array)

        # 4. Convert the BGR result to RGB for compatibility with PIL/matplotlib
        #    This is the operation from your snippet: [:, :, ::-1]
        print("Converting result from BGR to RGB...")
        src_image_seg_array_rgb = src_image_seg_array_bgr[:, :, ::-1]

        # 5. Create a Pillow (PIL) Image from the NumPy array
        #    This is the second operation from your snippet
        print("Creating PIL Image from array...")
        densepose_image = Image.fromarray(src_image_seg_array_rgb)
        
        # 6. Save the final segmentation image to disk
        densepose_image.save(output_path)
        print(f"✓ Successfully saved DensePose segmentation to: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        print("Please check your file paths and ensure all dependencies are installed correctly.")


if __name__ == "__main__":
    main()

import os
import numpy as np
from PIL import Image
import cv2
from leffa_utils.utils import resize_and_center
from preprocess.humanparsing.run_parsing import Parsing


class HumanParsingMaskGenerator:
    """
    A class to generate masks from human parsing results.
    It can produce binary (grayscale) masks for specific body parts
    or extract clothing parts while preserving their original colors.
    """
    def __init__(self, ckpt_dir="./ckpts", device="cuda"):
        self.device = device
        self.ckpt_dir = os.path.abspath(ckpt_dir)

        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        # Initialize the ONNX-based human parsing model
        self.parsing = Parsing(
            atr_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_atr.onnx"),
            lip_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_lip.onnx"),
        )

    def human_parser(self, src_image_pil):
        """
        Runs the human parsing model on a source image to get a label map.

        Args:
            src_image_pil (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The resulting label map where each pixel value
                             corresponds to a specific human part label.
        """
        # Resize and center the image to the model's expected input size
        image = resize_and_center(src_image_pil, 768, 1024)
        image_resized = image.resize((768, 1024)).convert("RGB")

        print("[🧠] Running human parsing...")
        label_map, _ = self.parsing(image_resized)
        label_map.save("label_map.png") # Save the raw label map for debugging

        return label_map

    def generate_mask(self, src_image_pil, target_labels=[12, 13, 14, 15, 18], fill_holes=True):
        """
        Generate a binary (grayscale) mask including only selected labels.

        Args:
            src_image_pil (PIL.Image.Image): The input image.
            target_labels (list): A list of integer labels to include in the mask.
                                  Defaults to skin and torso parts.
            fill_holes (bool): If True, applies a morphological closing operation
                               to fill small holes in the mask.

        Returns:
            PIL.Image.Image: A grayscale mask image.
        """
        print("[📸] Preparing input image for binary mask...")
        image = resize_and_center(src_image_pil, 768, 1024)
        # The parser can work on a slightly smaller image for speed
        image_resized = image.resize((384, 512)).convert("RGB")

        print("[🧠] Running human parsing...")
        label_map, _ = self.parsing(image_resized)
        
        # Convert PIL image to numpy array for processing
        if hasattr(label_map, "cpu"):
            label_map = label_map.cpu().numpy()
        if not isinstance(label_map, np.ndarray):
            label_map = np.array(label_map)

        print(f"[🎭] Generating binary mask for labels: {target_labels}")
        # Create a mask where pixels are 255 if their label is in target_labels, else 0
        binary_mask = np.isin(label_map, target_labels).astype(np.uint8) * 255

        # Resize mask to match the standard output size
        binary_mask = cv2.resize(binary_mask, (768, 1024), interpolation=cv2.INTER_NEAREST)

        if fill_holes:
            print("[🧽] Filling small holes in the mask...")
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(binary_mask, mode='L')

    def extract_clothing_with_original_colors(self, src_image_pil):
        """
        Generates an RGB mask showing only the clothing parts from the original image.
        The background is black, and the clothing retains its original color.
        Targets: upper_clothes, skirt, pants, dress, belt.

        Args:
            src_image_pil (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: An RGB image showing only the clothing.
        """
        print("[🎨] Extracting clothing with original colors...")
        
        # Ensure the source image is the correct size for pixel-perfect alignment
        image_resized_pil = resize_and_center(src_image_pil, 768, 1024)
        image_resized_np = np.array(image_resized_pil)

        # Get the full-resolution label map
        label_map_pil = self.human_parser(image_resized_pil)
        label_map_np = np.array(label_map_pil)

        # Define the labels for all clothing items
        # Labels: 4:upper_clothes, 5:skirt, 6:pants, 7:dress, 8:belt
        clothing_labels = [4, 5, 6, 7, 8]

        # Create a boolean mask that is True for any clothing pixel
        clothing_mask = np.isin(label_map_np, clothing_labels)

        # Create an empty (black) image with the same dimensions
        output_image_np = np.zeros_like(image_resized_np)

        # Where the mask is True, copy the pixels from the original image
        output_image_np[clothing_mask] = image_resized_np[clothing_mask]

        print("[✅] Clothing extraction complete.")
        return Image.fromarray(output_image_np, mode='RGB')

    def save_mask(self, image_pil, output_path="final_mask.png", labels=[12, 13, 14, 15, 18]):
        """Helper function to generate and save a binary mask."""
        mask = self.generate_mask(image_pil, target_labels=labels)
        mask.save(output_path)
        print(f"[✅] Binary mask saved to: {output_path}")
        return mask


if __name__ == "__main__":
    input_image_path = r"C:\Users\John123\Pictures\Screenshots\Screenshot 2025-07-09 232320.png"
    if not os.path.exists(input_image_path):
        print(f"[⚠️] '{input_image_path}' not found. Using a dummy image.")
        Image.new('RGB', (768, 1024), color='blue').save("dummy_image.png")
        input_image_path = "dummy_image.png"

    try:
        image = Image.open(input_image_path).convert("RGB")
        mask_generator = HumanParsingMaskGenerator(ckpt_dir="./ckpts", device="cpu")

        # --- Example 1: Save a binary (grayscale) mask for skin and torso ---
        print("\n--- Generating Binary Mask ---")
        mask_generator.save_mask(image, output_path="skin_torso_mask.png", labels=[12, 13, 14, 15, 18])

        # --- Example 2: Generate and save an image of the clothing with its original colors ---
        print("\n--- Extracting Original Clothing ---")
        clothing_image = mask_generator.extract_clothing_with_original_colors(image)
        clothing_image_path = "original_color_clothing_mask.png"
        clothing_image.save(clothing_image_path)
        print(f"[✅] Original color clothing mask saved to: {clothing_image_path}")


    except Exception as e:
        print(f"[❌] An error occurred: {e}")

import os
import numpy as np
from PIL import Image
import cv2
from leffa_utils.utils import resize_and_center
from preprocess.humanparsing.run_parsing import Parsing


class HumanParsingMaskGenerator:
    def __init__(self, ckpt_dir="./ckpts", device="cuda"):
        self.device = device
        self.ckpt_dir = os.path.abspath(ckpt_dir)

        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        self.parsing = Parsing(
            atr_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_atr.onnx"),
            lip_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_lip.onnx"),
        )
    def human_parser (self,src_image_pil):
        image = resize_and_center(src_image_pil, 768, 1024)
        image_resized = image.resize((768, 1024)).convert("RGB")

        print("[🧠] Running human parsing...")
        label_map, _ = self.parsing(image_resized)
        label_map.save("label_map.png")
        

        return label_map
    

    def generate_mask(self, src_image_pil, target_labels=[12, 13, 14, 15, 18], fill_holes=True):
        """Generate a binary mask including only selected labels (e.g., skin + torso)"""
        print("[📸] Preparing input image...")
        image = resize_and_center(src_image_pil, 768, 1024)
        image_resized = image.resize((384, 512)).convert("RGB")

        print("[🧠] Running human parsing...")
        label_map, _ = self.parsing(image_resized)
        label_map.save("label_map.png")
        if hasattr(label_map, "cpu"):
            label_map = label_map.cpu().numpy()
        if not isinstance(label_map, np.ndarray):
            label_map = np.array(label_map)

        print(f"[🎭] Generating mask for labels: {target_labels}")
        binary_mask = np.isin(label_map, target_labels).astype(np.uint8) * 255

        # Resize mask to match original size
        binary_mask = cv2.resize(binary_mask, (768, 1024), interpolation=cv2.INTER_NEAREST)

        if fill_holes:
            print("[🧽] Filling small holes in the mask...")
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(binary_mask, mode='L')

    def save_mask(self, image_pil, output_path="final_mask.png", labels=[12, 13, 14, 15, 18]):
        mask = self.generate_mask(image_pil, target_labels=labels)
        mask.save(output_path)
        print(f"[✅] Mask saved to: {output_path}")
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

        # Save torso + skin mask
        mask_generator.save_mask(image, output_path="skin_torso_mask.png", labels=[12, 13, 14, 18])

    except Exception as e:
        print(f"[❌] Error occurred: {e}")
          
          
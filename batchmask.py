import os
import cv2
import numpy as np
from PIL import Image

# This script now depends on your custom 'Parsing' class.
# Ensure the 'preprocess' directory is accessible from where you run this script.
try:
    from preprocess.humanparsing.run_parsing import Parsing
except ImportError:
    print("Error: Could not import 'Parsing' from 'preprocess.humanparsing.run_parsing'.")
    print("Please ensure the file exists and the 'preprocess' directory is in your Python path.")
    exit()

# --- HARDCODED CONFIGURATION ---
class Config:
    input_dir = r'A:\Poster\Newproject'
    
    # Folder for the transparent cutouts of the person
    person_output_dir = r'A:\Poster\person_cutouts' # Renamed for clarity

    atr_path = 'ckpts/humanparsing/parsing_atr.onnx'
    lip_path = 'ckpts/humanparsing/parsing_lip.onnx'
    
    # Dilation expands the mask slightly to prevent cropping the subject.
    dilation_amount = 1
    
    # --- Edge Feathering Configuration ---
    # Controls the softness of the cutout edge. Must be a positive odd number.
    # Higher numbers mean a softer, more feathered edge. Good values are 15, 21, 31.
    mask_feather_amount = 10

def generate_person_mask(model_parse: Image.Image, original_size: tuple, dilation: int, feather: int) -> Image.Image:
    """
    Generates a binary mask for the entire person and feathers the edges.
    """
    pred = np.array(model_parse)
    binary_mask_np = (pred > 0).astype(np.uint8) * 255
    
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary_mask_np = cv2.dilate(binary_mask_np, kernel, iterations=1)

    # Feather the mask edges by applying a Gaussian blur to the mask itself.
    if feather > 1:
        feather_kernel_size = (feather // 2) * 2 + 1
        binary_mask_np = cv2.GaussianBlur(binary_mask_np, (feather_kernel_size, feather_kernel_size), 0)

    # Convert the array to a PIL Image and resize to the original image dimensions.
    return Image.fromarray(binary_mask_np, 'L').resize(original_size, Image.NEAREST)


def apply_mask_for_cutout(original_image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Applies a binary mask to create a transparent PNG cutout of the person.
    """
    img_rgb = original_image.convert('RGB')
    # The feathered mask (with grayscale values) creates a semi-transparent edge.
    img_rgb.putalpha(mask.convert('L'))
    return img_rgb


def main(config):
    """
    Main function to extract the person from images and save the result.
    """
    print("Initializing Human Parsing model...")
    try:
        parsing_model = Parsing(atr_path=config.atr_path, lip_path=config.lip_path)
    except Exception as e:
        print(f"Failed to initialize Parsing model. Error: {e}")
        return

    os.makedirs(config.person_output_dir, exist_ok=True)
    
    print(f"Input folder: {config.input_dir}")
    print(f"Output folder for cutouts: {config.person_output_dir}")
    if config.mask_feather_amount > 1:
        print(f"Edge feather amount set to: {config.mask_feather_amount}")

    try:
        image_files = sorted([f for f in os.listdir(config.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    except FileNotFoundError:
        print(f"Error: Input directory not found at {config.input_dir}")
        return

    print(f"\nFound {len(image_files)} images to process...")

    for filename in image_files:
        input_path = os.path.join(config.input_dir, filename)
        base_filename = os.path.splitext(filename)[0]
        
        print(f"Processing: {filename}")
        
        try:
            source_image = Image.open(input_path).convert('RGB')
            model_parse_output, _ = parsing_model(source_image.resize((384, 512)))
            
            # 1. Generate the feathered binary mask for the entire person.
            person_mask = generate_person_mask(
                model_parse_output, 
                source_image.size, 
                config.dilation_amount,
                config.mask_feather_amount
            )
            
            # 2. Create the transparent person cutout.
            person_cutout = apply_mask_for_cutout(source_image, person_mask)
            
            # 3. Save the final transparent image. (Blur step is removed)
            output_path = os.path.join(config.person_output_dir, f"{base_filename}.png")
            person_cutout.save(output_path, 'PNG')
            
        except Exception as e:
            print(f"  Could not process {filename}. Error: {e}")

    print("\nProcessing complete!")


# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    config = Config()
    main(config)
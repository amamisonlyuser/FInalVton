import os
import cv2
import torch
import numpy as np
from numpy.linalg import lstsq
from PIL import Image, ImageDraw


def resize_and_center(image, target_width, target_height):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    original_height, original_width = img.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    resized_img = cv2.resize(img, (new_width, new_height),
                             interpolation=cv2.INTER_CUBIC)

    padded_img = np.ones((target_height, target_width, 3),
                         dtype=np.uint8) * 255

    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2

    padded_img[top:top + new_height, left:left + new_width] = resized_img

    return Image.fromarray(padded_img)


def list_dir(folder_path):
    # Collect all file paths within the directory
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    file_paths = sorted(file_paths)
    return file_paths


label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
    "neck": 18,
}


def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist


def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1,
                 mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask


def preprocess_garment_image(input_path, output_path=None, save_image=False):
    """
    Preprocess a garment image by cropping to a centered square,
    resizing, and pasting it onto a 768x1024 white background.
    """
    img = Image.open(input_path).convert('RGBA')
    
    # Step 1: Get the bounding box of the non-transparent pixels. (the garment)
    alpha = img.split()[-1]
    bbox = alpha.getbbox() # (left, upper, right, lower)
    if bbox is None:
        raise ValueError("No garment found in the image (the image may be fully transparent).")
    
    left, upper, right, lower = bbox
    bbox_width = right - left
    bbox_height = lower - upper
    
    # Step 2: Create a square crop that centers the garment.
    square_size = max(bbox_width, bbox_height)

    center_x = left + bbox_width // 2
    center_y = upper + bbox_height // 2

    new_left = center_x - square_size // 2
    new_upper = center_y - square_size // 2
    new_right = new_left + square_size
    new_lower = new_upper + square_size

    # Adjust the crop if it goes out of the image boundaries.
    if new_left < 0:
        new_left = 0
        new_right = square_size
    if new_upper < 0:
        new_upper = 0
        new_lower = square_size
    if new_right > img.width:
        new_right = img.width
        new_left = img.width - square_size
    if new_lower > img.height:
        new_lower = img.height
        new_upper = img.height - square_size

    # Crop the image to the computed square region.
    square_crop = img.crop((new_left, new_upper, new_right, new_lower))
    
    # Step 3: Resize the square crop.
    # Here we choose 768x768 so that it will occupy the full width when pasted.
    garment_resized = square_crop.resize((768, 768), Image.LANCZOS)
    
    # Step 4: Create a new white background image of 768x1024.
    background = Image.new('RGBA', (768, 1024), (255, 255, 255, 255))
    
    # Compute where to paste the resized garment so that it is centered.
    paste_x = 0
    paste_y = (1024 - 768) // 2
    
    # Paste the garment onto the background.
    background.paste(garment_resized, (paste_x, paste_y), garment_resized)
    
    # Optionally, convert to RGB (if you want to save as JPEG) or keep as PNG.
    final_image = background.convert("RGBA")
    
    if save_image:
        if output_path is None:
            raise ValueError("output_path must be provided if save_image is True.")
        final_image.save(output_path, "PNG")
    
    return final_image


def get_flexible_agnostic_mask(
    model_parse, 
    keypoint, 
    parts_to_mask: list, 
    size=(384, 512),
    # --- NEW PARAMETER ---
    # A positive integer expands the mask boundary, a negative integer shrinks it.
    boundary_scale: int = 0 
):
    """
    Generates a mask containing only the specified parts, with a final
    cleaning step and optional boundary scaling.
    """
    # --- (Previous code for mask generation remains the same) ---
    
    width, height = size
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    
    allowed_pixels_mask = np.zeros_like(parse_array, dtype=np.uint8)

    pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
    im_arms_left = Image.new('L', (width, height)); im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left); arms_draw_right = ImageDraw.Draw(im_arms_right)
    shoulder_right = np.multiply(pose_data[2][:2], height / 512.0); shoulder_left = np.multiply(pose_data[5][:2], height / 512.0)
    elbow_right = np.multiply(pose_data[3][:2], height / 512.0); elbow_left = np.multiply(pose_data[6][:2], height / 512.0)
    wrist_right = np.multiply(pose_data[4][:2], height / 512.0); wrist_left = np.multiply(pose_data[7][:2], height / 512.0)
    ARM_LINE_WIDTH = int(60 / 512 * height)

    if not (wrist_right[0] <= 1. and wrist_right[1] <= 1.):
        wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
        arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
    if not (wrist_left[0] <= 1. and wrist_left[1] <= 1.):
        wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
        arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

    drawn_arms_left = np.array(im_arms_left); drawn_arms_right = np.array(im_arms_right)
    parsed_arms_left = (parse_array == label_map["left_arm"]).astype(np.uint8); parsed_arms_right = (parse_array == label_map["right_arm"]).astype(np.uint8)
    hands_left = np.logical_and(parsed_arms_left, np.logical_not(drawn_arms_left)).astype(np.uint8) * 255
    hands_right = np.logical_and(parsed_arms_right, np.logical_not(drawn_arms_right)).astype(np.uint8) * 255
    
    for part in parts_to_mask:
        if part in label_map: allowed_pixels_mask[parse_array == label_map[part]] = 255
        elif part == "arms": allowed_pixels_mask |= drawn_arms_left; allowed_pixels_mask |= drawn_arms_right
        elif part == "hands": allowed_pixels_mask |= hands_left; allowed_pixels_mask |= hands_right
        elif part == "shoes": allowed_pixels_mask[parse_array == label_map["left_shoe"]] = 255; allowed_pixels_mask[parse_array == label_map["right_shoe"]] = 255
        else: print(f"Warning: Part '{part}' not recognized. Skipping.")
            
    processed_shape_mask = allowed_pixels_mask.copy()
    processed_shape_mask = hole_fill(processed_shape_mask)
    processed_shape_mask = refine_mask(processed_shape_mask)

    final_mask = cv2.bitwise_and(processed_shape_mask, allowed_pixels_mask)

    # --- NEW: Boundary Scaling Logic ---
    # This block applies dilation or erosion based on the boundary_scale parameter.
    if boundary_scale != 0:
        # The kernel size determines the amount of scaling.
        # It must be a positive odd integer.
        kernel_size = abs(boundary_scale) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if boundary_scale > 0:
            # Positive value expands (dilates) the white areas.
            print(f"✅ Expanding mask boundary with a {kernel_size}x{kernel_size} kernel.")
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        else: # boundary_scale < 0
            # Negative value shrinks (erodes) the white areas.
            print(f"Shrinking mask boundary with a {kernel_size}x{kernel_size} kernel.")
            final_mask = cv2.erode(final_mask, kernel, iterations=1)
            
    return Image.fromarray(final_mask)

def preserve_face_and_hair(
    original_model_image: Image.Image,
    virtual_tryon_image: Image.Image,
    model_parse: Image.Image
) -> Image.Image:
    """
    Extracts the face, hair, and neck from the original model image and 
    smoothly pastes them onto the virtual try-on image.

    This function helps in preserving the identity and fine details of the 
    model's face, which might get distorted during the try-on process.

    Args:
        original_model_image (Image.Image): The original image of the model.
        virtual_tryon_image (Image.Image): The output image from the virtual try-on pipeline.
        model_parse (Image.Image): The semantic segmentation map of the original model image.

    Returns:
        Image.Image: The virtual try-on image with the original, high-fidelity face,
                     hair, and neck blended in.
    """
    # Ensure all input images are in RGBA format for consistency
    original_model_image = original_model_image.convert("RGBA")
    virtual_tryon_image = virtual_tryon_image.convert("RGBA")
    
    # Resize the parse map to match the dimensions of the output image
    parse_map_resized = model_parse.resize(original_model_image.size, Image.NEAREST)
    parse_array = np.array(parse_map_resized)

    # 1. --- Create a Mask for the Face/Head Area ---
    face_part_labels = [
        label_map["hat"], label_map["sunglasses"], label_map["hair"], 
        label_map["head"]
    ]
    face_mask = np.zeros(parse_array.shape, dtype=np.uint8)
    for label in face_part_labels:
        face_mask[parse_array == label] = 255

    # 2. --- Create a Mask for Clothing and Subtract It --- ✨
    # Define labels for all possible clothing items
    clothing_part_labels = [
        label_map["upper_clothes"], label_map["dress"], label_map["scarf"]
    ]
    clothing_mask = np.zeros(parse_array.shape, dtype=np.uint8)
    for label in clothing_part_labels:
        clothing_mask[parse_array == label] = 255

    # Dilate the clothing mask to create a "buffer zone" around the clothes
    # This ensures we cleanly remove any clothing pixels near the neck/hair
    dilation_kernel = np.ones((5, 5), np.uint8)
    dilated_clothing_mask = cv2.dilate(clothing_mask, dilation_kernel, iterations=5)

    # Subtract the expanded clothing area from the face mask
    face_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(dilated_clothing_mask))

    # 3. --- Shrink and Feather the Final Mask for a Smooth Blend ---
    # Erode the mask to shrink it slightly, pulling the edges inward
    erosion_kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(face_mask, erosion_kernel, iterations=2)

    # Apply a Gaussian Blur to create soft, feathered edges
    # This is crucial for a seamless composite
    feathered_mask = cv2.GaussianBlur(eroded_mask, (15, 15), 0)

    # Convert the final numpy mask back to a PIL Image
    blend_mask = Image.fromarray(feathered_mask)

    # Composite the images using the refined mask
    final_image = Image.composite(original_model_image, virtual_tryon_image, blend_mask)

    return final_image.convert("RGB") # Return in RGB format

def get_skin_mask(
    model_parse: Image.Image,
    size=(384, 512),
    boundary_scale: int = 0
) -> Image.Image:
    """
    Generates a mask containing only the visible skin parts of the body,
    specifically excluding face, hair, and all clothing items.

    Args:
        model_parse (Image.Image): The semantic segmentation map of the model image.
        size (tuple): The target size (width, height) for the mask. Defaults to (384, 512).
        boundary_scale (int): A positive integer expands the mask boundary (dilation),
                              a negative integer shrinks it (erosion).

    Returns:
        Image.Image: A binary mask (white for skin, black for non-skin)
                     representing the visible skin areas.
    """
    width, height = size
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    # Initialize the skin mask to all zeros (black)
    skin_mask = np.zeros_like(parse_array, dtype=np.uint8)

    # Define labels that are *potentially* skin
    # We include head and neck here initially, and then explicitly remove face/hair.
    # This ensures we capture shoulders/upper chest if they are part of 'head' segmentation.
    potential_skin_labels = [
        label_map["left_leg"],
        label_map["right_leg"],
        label_map["left_arm"],
        label_map["right_arm"],
        label_map["neck"],
        label_map["head"] # Temporarily include 'head' to capture neck/shoulders
    ]

    # Define labels that are *definitely NOT* skin and must be excluded
    # This list is comprehensive for clothing, accessories, and face/hair.
    definite_non_skin_labels = [
        label_map["hat"],
        label_map["hair"],
        label_map["sunglasses"],
        label_map["upper_clothes"],
        label_map["skirt"],
        label_map["pants"],
        label_map["dress"],
        label_map["belt"],
        label_map["left_shoe"],
        label_map["right_shoe"],
        label_map["bag"],
        label_map["scarf"]
    ]

    # Add specific parts of 'head' that are not skin (like the face itself)
    # If your `head` label covers only the face, then including `head` here is correct.
    # If `head` includes neck/ears that should be skin, a more precise face segmentation
    # would be needed. Assuming `head` largely means the face area to be excluded.
    # The `preserve_face_and_hair` function handles the face, so we exclude it here.
    # We will rely on 'neck' for the neck area.
    definite_non_skin_labels.append(label_map["head"])


    # First pass: Mark all potential skin areas as white
    for label_value in potential_skin_labels:
        skin_mask[parse_array == label_value] = 255

    # Second pass: Explicitly remove all definite non-skin areas by setting them to black
    for label_value in definite_non_skin_labels:
        skin_mask[parse_array == label_value] = 0 # Overwrite any previous marking

    # Apply morphological operations for refinement
    # Hole filling helps close small gaps within the mask
    processed_skin_mask = hole_fill(skin_mask)
    # Refine mask helps select the largest connected component, removing small artifacts
    processed_skin_mask = refine_mask(processed_skin_mask)

    # Apply boundary scaling (dilation or erosion)
    if boundary_scale != 0:
        # Kernel size must be a positive odd integer for morphological operations
        kernel_size = abs(boundary_scale) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if boundary_scale > 0:
            # Positive scale: dilate (expand) the mask
            print(f"✅ Expanding skin mask boundary with a {kernel_size}x{kernel_size} kernel.")
            processed_skin_mask = cv2.dilate(processed_skin_mask, kernel, iterations=1)
        else:  # boundary_scale < 0
            # Negative scale: erode (shrink) the mask
            print(f"Shrinking skin mask boundary with a {kernel_size}x{kernel_size} kernel.")
            processed_skin_mask = cv2.erode(processed_skin_mask, kernel, iterations=1)

    return Image.fromarray(processed_skin_mask)
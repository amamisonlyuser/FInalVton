import torch
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import math
from typing import Tuple, Dict, Any

# Define a placeholder for nodes.MAX_RESOLUTION if it's not available in your environment
# This is typically a very large integer, e.g., 8192 or 16384
MAX_RESOLUTION = 8192 

class ComfyUIInpaintUtils:
    """
    A utility class to replicate the functionality of ComfyUI's
    InpaintCropImproved and InpaintStitchImproved nodes using PIL/NumPy.
    """

    def __init__(self):
        pass

    # --- Helper functions for image and mask manipulation ---

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a PyTorch Tensor (normalized to [0, 1])."""
        img_np = np.array(image).astype(np.float32) / 255.0
        if img_np.ndim == 2: # Grayscale mask
            img_np = img_np[None, :, :] # Add channel dimension
        else: # RGB image
            img_np = img_np[None, :, :, :].transpose(0, 3, 1, 2) # BHWC to BCHW
        return torch.from_numpy(img_np)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Converts a PyTorch Tensor to a PIL Image (denormalized to [0, 255])."""
        # Assuming tensor is BCHW or BHW for mask
        if tensor.ndim == 4: # Image
            img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # CHW to HWC
            img_np = (img_np * 255).astype(np.uint8)
            return Image.fromarray(img_np)
        elif tensor.ndim == 3: # Mask (BHW)
            mask_np = tensor.squeeze(0).cpu().numpy() # HW
            mask_np = (mask_np * 255).astype(np.uint8)
            return Image.fromarray(mask_np, mode='L')
        else:
            raise ValueError(f"Unsupported tensor dimensions: {tensor.ndim}")

    def _get_resample_filter(self, algorithm: str):
        """Maps algorithm string to PIL.Image.Resampling filter."""
        resample_map = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
        }
        return resample_map.get(algorithm, Image.Resampling.BICUBIC) # Default to BICUBIC

    def _resize_image(self, image: Image.Image, width: int, height: int, algorithm: str) -> Image.Image:
        """Resizes a PIL image using the specified algorithm."""
        return image.resize((width, height), self._get_resample_filter(algorithm))

    def _resize_mask(self, mask: Image.Image, width: int, height: int, algorithm: str) -> Image.Image:
        """Resizes a PIL mask (grayscale) using the specified algorithm (nearest is usually best for masks)."""
        # For masks, nearest neighbor is often preferred to preserve sharp edges
        return mask.resize((width, height), self._get_resample_filter(algorithm))

    def _fill_holes(self, mask: Image.Image) -> Image.Image:
        """Fills holes in a binary mask (PIL Image 'L' mode)."""
        # This is a basic approximation. For more robust hole filling,
        # scipy.ndimage.binary_fill_holes would be better but requires scipy.
        mask_np = np.array(mask)
        # Invert mask (holes become foreground)
        inverted_mask = ~mask_np
        # Use flood fill from border to mark external background
        from scipy.ndimage import binary_fill_holes
        filled_inverted = binary_fill_holes(inverted_mask)
        # Invert back to get filled holes
        filled_mask = ~filled_inverted
        return Image.fromarray(filled_mask.astype(np.uint8) * 255, mode='L')

    def _expand_mask(self, mask: Image.Image, pixels: int) -> Image.Image:
        """Expands a mask by a given number of pixels (dilation)."""
        if pixels <= 0:
            return mask
        # MaxFilter performs dilation for binary images
        return mask.filter(ImageFilter.MaxFilter(size=pixels * 2 + 1))

    def _invert_mask(self, mask: Image.Image) -> Image.Image:
        """Inverts a mask (0 becomes 255, 255 becomes 0)."""
        return ImageOps.invert(mask)

    def _blur_mask(self, mask: Image.Image, radius: float) -> Image.Image:
        """Blurs a mask using Gaussian blur."""
        if radius <= 0:
            return mask
        return mask.filter(ImageFilter.GaussianBlur(radius=radius))

    def _hipass_filter_mask(self, mask: Image.Image, threshold: float) -> Image.Image:
        """Applies a high-pass filter (thresholding) to a mask."""
        if threshold <= 0:
            return mask
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np[mask_np < threshold] = 0.0
        mask_np[mask_np >= threshold] = 1.0 # Binarize after threshold
        return Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

    def _extend_image_and_mask(self, image: Image.Image, mask: Image.Image, optional_context_mask: Image.Image,
                                up_factor: float, down_factor: float, left_factor: float, right_factor: float) -> Tuple[Image.Image, Image.Image, Image.Image, Tuple[int, int, int, int]]:
        """
        Extends the image and masks for outpainting.
        Returns extended image, extended mask, extended optional_context_mask, and the bbox of original content.
        """
        w, h = image.size
        # Calculate new dimensions
        new_w = int(w * (1 + left_factor + right_factor))
        new_h = int(h * (1 + up_factor + down_factor))

        # Calculate offsets for pasting original content
        offset_x = int(w * left_factor)
        offset_y = int(h * up_factor)

        # Create new blank canvas images/masks
        extended_image = Image.new(image.mode, (new_w, new_h), color='black') # Or a neutral color
        extended_mask = Image.new('L', (new_w, new_h), color=0) # Black for mask (unmasked)
        extended_optional_context_mask = Image.new('L', (new_w, new_h), color=0)

        # Paste original content into the center of the new canvas
        extended_image.paste(image, (offset_x, offset_y))
        extended_mask.paste(mask, (offset_x, offset_y))
        extended_optional_context_mask.paste(optional_context_mask, (offset_x, offset_y))

        # Return the bounding box of the original content within the new extended image
        original_bbox_in_extended = (offset_x, offset_y, w, h)

        return extended_image, extended_mask, extended_optional_context_mask, original_bbox_in_extended

    def _get_mask_bbox(self, mask: Image.Image) -> Tuple[int, int, int, int]:
        """
        Calculates the bounding box (x, y, width, height) of non-zero pixels in a mask.
        Returns (-1, -1, -1, -1) if the mask is entirely black.
        """
        mask_np = np.array(mask)
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return -1, -1, -1, -1 # Mask is empty

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1)

    def _grow_bbox(self, bbox: Tuple[int, int, int, int], factor: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Grows a bounding box by a given factor, clamping to image boundaries."""
        x, y, w, h = bbox
        
        if x == -1: # Empty mask case
            return 0, 0, img_w, img_h

        # Calculate new dimensions
        new_w = int(w * factor)
        new_h = int(h * factor)

        # Calculate new top-left corner
        new_x = x - (new_w - w) // 2
        new_y = y - (new_h - h) // 2

        # Clamp to image boundaries
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)
        
        # Recalculate width/height based on clamped coordinates
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)

        return new_x, new_y, new_w, new_h

    def _crop_and_resize(self, image: Image.Image, mask: Image.Image, bbox: Tuple[int, int, int, int],
                         target_width: int, target_height: int, padding: int,
                         downscale_algo: str, upscale_algo: str) -> Tuple[Image.Image, Image.Image, Dict]:
        """
        Crops the image and mask based on the bounding box, adds padding,
        and resizes to target dimensions.
        Returns cropped_image, cropped_mask, and stitcher info.
        """
        x, y, w, h = bbox
        
        if w <= 0 or h <= 0: # Handle empty or invalid bbox
            # Return a black image/mask of target size if no valid crop area
            cropped_image = Image.new(image.mode, (target_width, target_height), color='black')
            cropped_mask = Image.new('L', (target_width, target_height), color=0)
            stitcher_info = {
                'canvas_to_orig_x': 0, 'canvas_to_orig_y': 0, 'canvas_to_orig_w': 0, 'canvas_to_orig_h': 0,
                'canvas_image': Image.new(image.mode, image.size, color='black'), # Empty canvas
                'cropped_to_canvas_x': 0, 'cropped_to_canvas_y': 0, 'cropped_to_canvas_w': target_width, 'cropped_to_canvas_h': target_height,
                'cropped_mask_for_blend': Image.new('L', (target_width, target_height), color=0),
                'downscale_algorithm': downscale_algo, 'upscale_algorithm': upscale_algo, 'blend_pixels': 0 # No blend if no mask
            }
            return cropped_image, cropped_mask, stitcher_info

        # Add padding to the bounding box
        padded_x = max(0, x - padding)
        padded_y = max(0, y - padding)
        padded_w = min(w + 2 * padding, image.width - padded_x)
        padded_h = min(h + 2 * padding, image.height - padded_y)

        # Crop the image and mask
        cropped_image_raw = image.crop((padded_x, padded_y, padded_x + padded_w, padded_y + padded_h))
        cropped_mask_raw = mask.crop((padded_x, padded_y, padded_x + padded_w, padded_y + padded_h))

        # Resize cropped image and mask to target dimensions
        cropped_image = self._resize_image(cropped_image_raw, target_width, target_height, upscale_algo)
        cropped_mask = self._resize_mask(cropped_mask_raw, target_width, target_height, self._get_resample_filter(downscale_algo)) # Use downscale_algo for mask resize? Or nearest? ComfyUI uses bilinear for downscale, bicubic for upscale. For masks, nearest is often safer.

        # Prepare stitcher information
        # This represents the cropped area's position and size relative to the *original* image
        # and its position and size relative to the *cropped* image (which is now target_width/height)
        stitcher_info = {
            'canvas_to_orig_x': padded_x,
            'canvas_to_orig_y': padded_y,
            'canvas_to_orig_w': padded_w,
            'canvas_to_orig_h': padded_h,
            'canvas_image': image.copy(), # Store original image for stitching
            'cropped_to_canvas_x': 0, # Cropped image starts at 0,0 within its own canvas
            'cropped_to_canvas_y': 0,
            'cropped_to_canvas_w': target_width,
            'cropped_to_canvas_h': target_height,
            'cropped_mask_for_blend': cropped_mask.copy(), # Mask for blending
            'downscale_algorithm': downscale_algo,
            'upscale_algorithm': upscale_algo,
            'blend_pixels': padding # Using padding as blend_pixels for simplicity, as per ComfyUI node's default
        }
        return cropped_image, cropped_mask, stitcher_info

    def _stitch_image(self, stitcher: Dict[str, Any], inpainted_cropped_image: Image.Image) -> Image.Image:
        """
        Stitches the inpainted (cropped) image back into the original image.
        """
        original_image = stitcher['canvas_image']
        
        # Resize inpainted_cropped_image back to the original cropped region's size
        # for blending, using the upscale algorithm
        resized_inpainted_cropped = self._resize_image(
            inpainted_cropped_image,
            stitcher['cropped_to_canvas_w'], # This should be the target_width from crop
            stitcher['cropped_to_canvas_h'], # This should be the target_height from crop
            stitcher['upscale_algorithm']
        )
        
        # The mask used for blending is the `cropped_mask_for_blend` which was resized to target size
        # and potentially blurred during pre-processing.
        blend_mask = stitcher['cropped_mask_for_blend']
        
        # Ensure blend_mask is the same size as resized_inpainted_cropped
        if blend_mask.size != resized_inpainted_cropped.size:
             blend_mask = self._resize_mask(blend_mask, resized_inpainted_cropped.width, resized_inpainted_cropped.height, "nearest")


        # Create a canvas to paste the inpainted image onto, which matches the size
        # of the original cropped region (padded_w, padded_h)
        canvas_for_paste = Image.new(original_image.mode, (stitcher['canvas_to_orig_w'], stitcher['canvas_to_orig_h']), color='black')
        
        # Paste the resized inpainted image onto this canvas at its original cropped position (0,0)
        canvas_for_paste.paste(resized_inpainted_cropped, (0, 0), blend_mask if blend_mask.mode == 'L' else None) # Use mask for alpha blending if available

        # Now, paste this canvas back onto the original image
        final_image = original_image.copy()
        final_image.paste(canvas_for_paste, (stitcher['canvas_to_orig_x'], stitcher['canvas_to_orig_y']))

        return final_image

    # --- Main Pre-processing and Post-processing methods ---

    def preprocess_for_inpaint(self, image_pil: Image.Image, mask_pil: Image.Image, optional_context_mask_pil: Image.Image = None,
                                downscale_algorithm: str = "bilinear", upscale_algorithm: str = "bicubic",
                                preresize: bool = False, preresize_mode: str = "ensure minimum resolution",
                                preresize_min_width: int = 1024, preresize_min_height: int = 1024,
                                preresize_max_width: int = MAX_RESOLUTION, preresize_max_height: int = MAX_RESOLUTION,
                                mask_fill_holes: bool = True, mask_expand_pixels: int = 0, mask_invert: bool = False,
                                mask_blend_pixels: int = 32, mask_hipass_filter: float = 0.1,
                                extend_for_outpainting: bool = False,
                                extend_up_factor: float = 1.0, extend_down_factor: float = 1.0,
                                extend_left_factor: float = 1.0, extend_right_factor: float = 1.0,
                                context_from_mask_extend_factor: float = 2.0,
                                output_resize_to_target_size: bool = True,
                                output_target_width: int = 512, output_target_height: int = 512,
                                output_padding: int = 32) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Replicates InpaintCropImproved logic.
        Processes the input image and mask to prepare them for an inpainting model.
        Returns the cropped image, cropped mask (as PyTorch Tensors), and stitcher info.
        """
        image = image_pil.convert("RGB")
        mask = mask_pil.convert("L") # Ensure mask is grayscale
        optional_context_mask = optional_context_mask_pil.convert("L") if optional_context_mask_pil else Image.new('L', image.size, 0)

        # 1. Preresize (if enabled)
        if preresize:
            # This is a simplified preresize. ComfyUI's preresize_imm is more complex.
            # For simplicity, we'll just resize to min_width/height if smaller.
            current_w, current_h = image.size
            target_w, target_h = current_w, current_h

            if preresize_mode == "ensure minimum resolution" or preresize_mode == "ensure minimum and maximum resolution":
                if current_w < preresize_min_width or current_h < preresize_min_height:
                    scale = max(preresize_min_width / current_w, preresize_min_height / current_h)
                    target_w, target_h = int(current_w * scale), int(current_h * scale)
            
            if preresize_mode == "ensure maximum resolution" or preresize_mode == "ensure minimum and maximum resolution":
                if current_w > preresize_max_width or current_h > preresize_max_height:
                    scale = min(preresize_max_width / current_w, preresize_max_height / current_h)
                    target_w, target_h = int(current_w * scale), int(current_h * scale)

            if target_w != current_w or target_h != current_h:
                image = self._resize_image(image, target_w, target_h, downscale_algorithm)
                mask = self._resize_mask(mask, target_w, target_h, downscale_algorithm)
                optional_context_mask = self._resize_mask(optional_context_mask, target_w, target_h, downscale_algorithm)

        # 2. Mask manipulation
        if mask_fill_holes:
            mask = self._fill_holes(mask)
        if mask_expand_pixels > 0:
            mask = self._expand_mask(mask, mask_expand_pixels)
        if mask_invert:
            mask = self._invert_mask(mask)
        if mask_blend_pixels > 0:
            # ComfyUI applies blur *after* expand for blend.
            # We already expanded by mask_expand_pixels.
            # If mask_blend_pixels is also > 0, we need to expand/blur for blending.
            # For simplicity, we'll use mask_blend_pixels as the blur radius here.
            mask = self._blur_mask(mask, mask_blend_pixels * 0.5) # Approximate blur radius

        if mask_hipass_filter >= 0.01:
            mask = self._hipass_filter_mask(mask, mask_hipass_filter)
            optional_context_mask = self._hipass_filter_mask(optional_context_mask, mask_hipass_filter)

        # 3. Extend for outpainting
        original_bbox_in_extended = (0, 0, image.width, image.height) # Default if not extending
        if extend_for_outpainting:
            image, mask, optional_context_mask, original_bbox_in_extended = self._extend_image_and_mask(
                image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor
            )

        # 4. Context area calculation
        bbox = self._get_mask_bbox(mask)
        if bbox[0] == -1: # If mask is empty, use full image as context
            bbox = (0, 0, image.width, image.height)

        if context_from_mask_extend_factor >= 1.01:
            bbox = self._grow_bbox(bbox, context_from_mask_extend_factor, image.width, image.height)

        # Combine with optional_context_mask (if any, expand bbox to include it)
        if optional_context_mask_pil:
            opt_mask_bbox = self._get_mask_bbox(optional_context_mask)
            if opt_mask_bbox[0] != -1: # If optional context mask is not empty
                # Combine bounding boxes
                min_x = min(bbox[0], opt_mask_bbox[0])
                min_y = min(bbox[1], opt_mask_bbox[1])
                max_x = max(bbox[0] + bbox[2], opt_mask_bbox[0] + opt_mask_bbox[2])
                max_y = max(bbox[1] + bbox[3], opt_mask_bbox[1] + opt_mask_bbox[3])
                bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
                # Clamp combined bbox to image boundaries
                bbox = (max(0, bbox[0]), max(0, bbox[1]),
                        min(bbox[2], image.width - max(0, bbox[0])),
                        min(bbox[3], image.height - max(0, bbox[1])))


        # 5. Crop and resize to target output
        cropped_image_pil, cropped_mask_pil, stitcher_info = self._crop_and_resize(
            image, mask, bbox, output_target_width, output_target_height, output_padding,
            downscale_algorithm, upscale_algorithm
        )
        
        # Store original_bbox_in_extended for stitching if outpainting was done
        stitcher_info['original_bbox_in_extended'] = original_bbox_in_extended

        # Convert to PyTorch Tensors for diffusers pipeline
        cropped_image_tensor = self._pil_to_tensor(cropped_image_pil)
        cropped_mask_tensor = self._pil_to_tensor(cropped_mask_pil)

        return cropped_image_tensor, cropped_mask_tensor, stitcher_info

    def postprocess_inpainted_image(self, stitcher_info: Dict[str, Any], inpainted_image_tensor: torch.Tensor) -> Image.Image:
        """
        Replicates InpaintStitchImproved logic.
        Stitches the inpainted image back into the original image.
        Returns the final stitched PIL Image.
        """
        inpainted_image_pil = self._tensor_to_pil(inpainted_image_tensor)
        final_image = self._stitch_image(stitcher_info, inpainted_image_pil)
        return final_image


# --- Example Usage in a Hypothetical SD Workflow ---
# (This part is for demonstration and not part of the compact utility class itself)

# from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
# from diffusers.utils import load_image
# import os

# # Assuming you have a pipeline initialized like this:
# # pipeline = StableDiffusionInpaintPipeline.from_single_file(
# #     "path/to/your/inpainting_model.safetensors", torch_dtype=torch.float16
# # )
# # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
# # pipeline.to("cuda")
# # pipeline.load_ip_adapter(...) # If using IP-Adapter

# # Example input images (replace with your actual PIL Images)
# # original_image_pil = Image.open("Screenshot 2025-06-06 051538.png").convert("RGB")
# # mask_image_pil = Image.open("skin_torso_mask.png").convert("L")
# # ip_adapter_image_pil = Image.open("image1 (3).png").convert("RGB")
# # optional_context_mask_pil = None # Or load another mask if needed

# # Example parameters (matching your ComfyUI workflow)
# # params = {
# #     "downscale_algorithm": "bilinear",
# #     "upscale_algorithm": "bicubic",
# #     "preresize": False,
# #     "preresize_mode": "ensure minimum resolution",
# #     "preresize_min_width": 1024,
# #     "preresize_min_height": 1024,
# #     "preresize_max_width": MAX_RESOLUTION,
# #     "preresize_max_height": MAX_RESOLUTION,
# #     "mask_fill_holes": True,
# #     "mask_expand_pixels": 0,
# #     "mask_invert": False,
# #     "mask_blend_pixels": 32,
# #     "mask_hipass_filter": 0.1,
# #     "extend_for_outpainting": False,
# #     "extend_up_factor": 1.0,
# #     "extend_down_factor": 1.0,
# #     "extend_left_factor": 1.0,
# #     "extend_right_factor": 1.0,
# #     "context_from_mask_extend_factor": 2.0000000000000004, # From your workflow
# #     "output_resize_to_target_size": True,
# #     "output_target_width": 512,
# #     "output_target_height": 512,
# #     "output_padding": 32, # Converted from string "32"
# # }

# # Initialize the utility
# # inpaint_utils = ComfyUIInpaintUtils()

# # --- Pre-processing ---
# # cropped_image_tensor, cropped_mask_tensor, stitcher_info = inpaint_utils.preprocess_for_inpaint(
# #     image_pil=original_image_pil,
# #     mask_pil=mask_image_pil,
# #     optional_context_mask_pil=optional_context_mask_pil,
# #     **params
# # )

# # --- Run Inpainting (using your existing diffusers pipeline logic) ---
# # generator = torch.Generator(device="cuda").manual_seed(39163971096256)
# # inpainted_cropped_tensor = pipeline(
# #     prompt="body, skin, hands, torso, legs, shoulder",
# #     negative_prompt="ugly, text, watermark",
# #     image=cropped_image_tensor,
# #     mask_image=cropped_mask_tensor,
# #     ip_adapter_image=ip_adapter_image_pil, # This needs to be a PIL image for pipeline input
# #     num_inference_steps=20,
# #     guidance_scale=8.0,
# #     generator=generator,
# #     strength=1.0,
# # ).images[0] # Get the first image from the list

# # --- Post-processing ---
# # final_stitched_image_pil = inpaint_utils.postprocess_inpainted_image(
# #     stitcher_info,
# #     inpainted_cropped_tensor
# # )

# # final_stitched_image_pil.save("final_output_stitched.png")
# # print("Final stitched image saved!")


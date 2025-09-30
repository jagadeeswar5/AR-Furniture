# Background Inpainting Furniture Replacement
# This approach fills the masked area with background content first, then places furniture

import cv2
import numpy as np
import os

def background_inpainting_replacement(original_image, mask_np, suggested_furniture, thumbnail_folder):
    """
    Background Inpainting Approach:
    1. Use cv2.inpaint to intelligently fill the masked area with background content
    2. Place furniture on the clean inpainted background
    3. This ensures the original object is completely removed first
    """
    try:
        print("Starting background inpainting replacement...")
        
        # Step 1: Create a clean binary mask for inpainting
        binary_mask = (mask_np > 127).astype(np.uint8) * 255
        
        print(f"Mask shape: {binary_mask.shape}")
        print(f"Original image shape: {original_image.shape}")
        
        # Step 2: Use cv2.inpaint to fill the masked area with background content
        # INPAINT_TELEA is good for removing objects and filling with background
        inpaint_radius = 3  # Small radius for precise inpainting
        inpainted_image = cv2.inpaint(original_image, binary_mask, inpaint_radius, cv2.INPAINT_TELEA)
        
        print("✅ Background inpainting completed")
        
        # Step 3: Get furniture image
        furniture_img_path = os.path.join(thumbnail_folder, suggested_furniture + ".png")
        if not os.path.exists(furniture_img_path):
            raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
        
        furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError("Could not load furniture image")
        
        # Step 4: Get mask bounds for furniture placement
        y_indices, x_indices = np.where(binary_mask > 127)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("No valid mask found")
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        
        # Step 5: Scale furniture to fit the mask area
        furniture_h, furniture_w = furniture_image.shape[:2]
        if furniture_w == 0 or furniture_h == 0:
            raise ValueError("Invalid furniture image dimensions")
        
        scale = min(mask_width / furniture_w, mask_height / furniture_h)
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        print(f"Furniture scale: {scale:.3f}, new size: {new_w}x{new_h}")
        
        # Step 6: Resize furniture
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Step 7: Center furniture within mask bounds
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)
        
        print(f"Furniture placement: x={offset_x}, y={offset_y}")
        
        # Step 8: Place furniture on the inpainted background
        result_image = inpainted_image.copy()
        
        # Create a mask for the furniture placement area
        mask_roi = binary_mask[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        
        # Replace pixels where mask exists with furniture pixels
        for y in range(new_h):
            for x in range(new_w):
                if (y + offset_y < result_image.shape[0] and 
                    x + offset_x < result_image.shape[1] and
                    mask_roi[y, x] > 127):  # If mask pixel is selected
                    
                    if furniture_resized.shape[2] == 4:  # Has alpha channel
                        # Use furniture pixel if it's not transparent
                        if furniture_resized[y, x, 3] > 0:
                            result_image[offset_y + y, offset_x + x] = furniture_resized[y, x, :3]
                    else:  # No alpha channel
                        # Direct replacement
                        result_image[offset_y + y, offset_x + x] = furniture_resized[y, x, :3]
        
        print("✅ Furniture placement on inpainted background completed")
        return result_image
        
    except Exception as e:
        print(f"Background inpainting replacement failed: {e}")
        return original_image.copy()

def content_aware_replacement(original_image, mask_np, suggested_furniture, thumbnail_folder):
    """
    Content-Aware Fill Approach:
    Uses cv2.INPAINT_NS (Navier-Stokes) for better content understanding
    """
    try:
        print("Starting content-aware replacement...")
        
        # Create binary mask
        binary_mask = (mask_np > 127).astype(np.uint8) * 255
        
        # Use Navier-Stokes inpainting for better content understanding
        inpainted_image = cv2.inpaint(original_image, binary_mask, 10, cv2.INPAINT_NS)
        
        print("✅ Content-aware inpainting completed")
        
        # Rest is same as background inpainting approach
        furniture_img_path = os.path.join(thumbnail_folder, suggested_furniture + ".png")
        if not os.path.exists(furniture_img_path):
            raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
        
        furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError("Could not load furniture image")
        
        y_indices, x_indices = np.where(binary_mask > 127)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        
        furniture_h, furniture_w = furniture_image.shape[:2]
        scale = min(mask_width / furniture_w, mask_height / furniture_h)
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)
        
        result_image = inpainted_image.copy()
        mask_roi = binary_mask[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        
        for y in range(new_h):
            for x in range(new_w):
                if (y + offset_y < result_image.shape[0] and 
                    x + offset_x < result_image.shape[1] and
                    mask_roi[y, x] > 127):
                    
                    if furniture_resized.shape[2] == 4:
                        if furniture_resized[y, x, 3] > 0:
                            result_image[offset_y + y, offset_x + x] = furniture_resized[y, x, :3]
                    else:
                        result_image[offset_y + y, offset_x + x] = furniture_resized[y, x, :3]
        
        print("✅ Content-aware furniture placement completed")
        return result_image
        
    except Exception as e:
        print(f"Content-aware replacement failed: {e}")
        return original_image.copy()

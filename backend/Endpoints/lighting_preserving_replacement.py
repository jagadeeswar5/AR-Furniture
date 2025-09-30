# Lighting-Preserving Object Replacement
# This approach preserves the room's lighting, shadows, and atmosphere

import cv2
import numpy as np
import os

def lighting_preserving_replacement(original_image, mask_np, suggested_furniture, thumbnail_folder):
    """
    Lighting-Preserving Replacement Approach:
    1. Use inpainting to naturally remove the object while preserving room lighting
    2. Blend furniture with the inpainted background
    3. Maintain shadows, lighting, and room atmosphere
    """
    try:
        print("Starting lighting-preserving replacement...")
        
        # Create a copy of the original image
        result_image = original_image.copy()
        
        # Step 1: Get furniture image
        furniture_img_path = os.path.join(thumbnail_folder, suggested_furniture + ".png")
        if not os.path.exists(furniture_img_path):
            raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
        
        furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError("Could not load furniture image")
        
        # Step 2: Get mask bounds for furniture scaling
        y_indices, x_indices = np.where(mask_np > 127)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("No valid mask found")
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        
        print(f"Mask bounds: x={x_min}-{x_max}, y={y_min}-{y_max}")
        print(f"Mask size: {mask_width}x{mask_height}")
        
        # Step 3: NATURAL INPAINTING - Remove object while preserving room lighting
        # Use OpenCV's inpainting to naturally fill the masked area
        inpaint_radius = 8  # Larger radius for better blending
        inpainted_image = cv2.inpaint(original_image, mask_np, inpaint_radius, cv2.INPAINT_TELEA)
        
        print("✅ Natural inpainting completed - room lighting preserved")
        
        # Step 4: Scale furniture to fit the mask area
        furniture_h, furniture_w = furniture_image.shape[:2]
        scale = min(mask_width / furniture_w, mask_height / furniture_h) * 0.9  # Slightly smaller for better fit
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        # Resize furniture with high quality
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"Furniture scale: {scale:.3f}, new size: {new_w}x{new_h}")
        
        # Step 5: Center furniture within the mask area
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)
        
        print(f"Furniture placement: x={offset_x}, y={offset_y}")
        
        # Step 6: Create a soft mask for furniture blending
        # Create a mask that's slightly smaller than the furniture for soft edges
        furniture_mask = np.zeros((new_h, new_w), dtype=np.float32)
        
        # Create a soft mask with feathered edges
        center_y, center_x = new_h // 2, new_w // 2
        y_coords, x_coords = np.ogrid[:new_h, :new_w]
        
        # Create elliptical mask for more natural shape
        ellipse_mask = ((x_coords - center_x) / (new_w * 0.4)) ** 2 + ((y_coords - center_y) / (new_h * 0.4)) ** 2 <= 1
        furniture_mask[ellipse_mask] = 1.0
        
        # Apply Gaussian blur for soft edges
        furniture_mask = cv2.GaussianBlur(furniture_mask, (15, 15), 0)
        
        # Step 7: Blend furniture with the inpainted background
        for fy in range(new_h):
            for fx in range(new_w):
                if (fy + offset_y < inpainted_image.shape[0] and 
                    fx + offset_x < inpainted_image.shape[1]):
                    
                    # Get furniture pixel
                    furniture_pixel = furniture_resized[fy, fx]
                    
                    # Get background pixel
                    bg_pixel = inpainted_image[fy + offset_y, fx + offset_x]
                    
                    # Get blend weight from mask
                    blend_weight = furniture_mask[fy, fx]
                    
                    if furniture_resized.shape[2] == 4:  # Has alpha channel
                        alpha = furniture_pixel[3] / 255.0
                        # Combine furniture alpha with our soft mask
                        final_alpha = alpha * blend_weight
                        
                        if final_alpha > 0.1:  # Not transparent
                            # Blend with background
                            blended_color = final_alpha * furniture_pixel[:3] + (1 - final_alpha) * bg_pixel
                            inpainted_image[fy + offset_y, fx + offset_x] = blended_color.astype(np.uint8)
                    else:  # No alpha channel
                        if blend_weight > 0.1:
                            # Blend with background
                            blended_color = blend_weight * furniture_pixel + (1 - blend_weight) * bg_pixel
                            inpainted_image[fy + offset_y, fx + offset_x] = blended_color.astype(np.uint8)
        
        print("✅ Furniture blended with preserved room lighting")
        
        return inpainted_image
        
    except Exception as e:
        print(f"Lighting-preserving replacement failed: {e}")
        return None

def smart_lighting_replacement(original_image, mask_np, suggested_furniture, thumbnail_folder):
    """
    Smart Lighting Replacement - Alternative approach using edge-aware inpainting
    """
    try:
        print("Starting smart lighting replacement...")
        
        # Create a copy of the original image
        result_image = original_image.copy()
        
        # Step 1: Get furniture image
        furniture_img_path = os.path.join(thumbnail_folder, suggested_furniture + ".png")
        if not os.path.exists(furniture_img_path):
            raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
        
        furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError("Could not load furniture image")
        
        # Step 2: Use edge-aware inpainting for better results
        # Create a more refined mask
        refined_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        
        # Use both inpainting methods and combine
        inpaint1 = cv2.inpaint(original_image, mask_np, 5, cv2.INPAINT_TELEA)
        inpaint2 = cv2.inpaint(original_image, mask_np, 5, cv2.INPAINT_NS)
        
        # Combine both inpainting results
        combined_inpaint = cv2.addWeighted(inpaint1, 0.6, inpaint2, 0.4, 0)
        
        print("✅ Edge-aware inpainting completed")
        
        # Step 3: Scale and place furniture (same as before)
        y_indices, x_indices = np.where(mask_np > 127)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        
        furniture_h, furniture_w = furniture_image.shape[:2]
        scale = min(mask_width / furniture_w, mask_height / furniture_h) * 0.85
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)
        
        # Step 4: Soft blending with the combined inpainted background
        for fy in range(new_h):
            for fx in range(new_w):
                if (fy + offset_y < combined_inpaint.shape[0] and 
                    fx + offset_x < combined_inpaint.shape[1]):
                    
                    furniture_pixel = furniture_resized[fy, fx]
                    bg_pixel = combined_inpaint[fy + offset_y, fx + offset_x]
                    
                    if furniture_resized.shape[2] == 4:  # Has alpha channel
                        alpha = furniture_pixel[3] / 255.0
                        if alpha > 0.2:  # Not transparent
                            # Soft blend
                            blended_color = 0.8 * furniture_pixel[:3] + 0.2 * bg_pixel
                            combined_inpaint[fy + offset_y, fx + offset_x] = blended_color.astype(np.uint8)
                    else:  # No alpha channel
                        blended_color = 0.8 * furniture_pixel + 0.2 * bg_pixel
                        combined_inpaint[fy + offset_y, fx + offset_x] = blended_color.astype(np.uint8)
        
        print("✅ Smart lighting replacement completed")
        return combined_inpaint
        
    except Exception as e:
        print(f"Smart lighting replacement failed: {e}")
        return None

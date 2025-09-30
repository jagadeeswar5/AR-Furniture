# Simple Furniture Replacement Function
# This replaces the complex inpainting logic with a dead simple approach

def simple_furniture_replacement(original_image, mask_np, suggested_furniture, thumbnail_folder):
    """
    Dead simple furniture replacement:
    1. Scale furniture to fit mask
    2. Replace masked pixels with furniture pixels
    3. Done - no AI, no complex blending, no distortion
    """
    import cv2
    import numpy as np
    import os
    
    try:
        # Get furniture image
        furniture_img_path = os.path.join(thumbnail_folder, suggested_furniture + ".png")
        if not os.path.exists(furniture_img_path):
            raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
        
        furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError("Could not load furniture image")
        
        # Get mask bounds
        y_indices, x_indices = np.where(mask_np > 127)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("No valid mask found")
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        
        # Get furniture dimensions
        furniture_h, furniture_w = furniture_image.shape[:2]
        if furniture_w == 0 or furniture_h == 0:
            raise ValueError("Invalid furniture image dimensions")
        
        # Scale furniture to fit mask while preserving aspect ratio
        scale = min(mask_width / furniture_w, mask_height / furniture_h)
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        # Resize furniture
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center furniture within mask
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)
        
        # Create copy of original image
        result_image = original_image.copy()
        
        # Simple replacement: where mask exists, use furniture
        mask_roi = mask_np[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        
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
        
        return result_image
        
    except Exception as e:
        print(f"Simple replacement failed: {e}")
        return original_image.copy()

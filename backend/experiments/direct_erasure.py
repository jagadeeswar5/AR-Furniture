# Direct Object Erasure + Furniture Placement
# This approach directly erases the masked area and places furniture without any AI processing

import cv2
import numpy as np
import os

def direct_erasure_replacement(original_image, mask_np, suggested_furniture, thumbnail_folder):
    """
    Direct Erasure Approach:
    1. Directly erase the masked area (fill with white/transparent)
    2. Place furniture on the erased area
    3. No AI processing, no distortion, just simple replacement
    """
    try:
        print("Starting direct erasure replacement...")
        
        # Create a copy of the original image
        result_image = original_image.copy()
        
        # Step 1: Get furniture image
        furniture_img_path = os.path.join(thumbnail_folder, suggested_furniture + ".png")
        if not os.path.exists(furniture_img_path):
            raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
        
        furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError("Could not load furniture image")
        
        # Step 2: Get mask bounds
        y_indices, x_indices = np.where(mask_np > 127)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("No valid mask found")
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        
        print(f"Mask bounds: x={x_min}-{x_max}, y={y_min}-{y_max}")
        print(f"Mask size: {mask_width}x{mask_height}")
        
        # Step 3: Get furniture dimensions and scale
        furniture_h, furniture_w = furniture_image.shape[:2]
        if furniture_w == 0 or furniture_h == 0:
            raise ValueError("Invalid furniture image dimensions")
        
        # Scale furniture to fit the mask area while preserving aspect ratio
        scale = min(mask_width / furniture_w, mask_height / furniture_h)
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        print(f"Furniture scale: {scale:.3f}, new size: {new_w}x{new_h}")
        
        # Step 4: Resize furniture
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Step 5: Center furniture within mask bounds
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)
        
        print(f"Furniture placement: x={offset_x}, y={offset_y}")
        
        # Step 6: DIRECT ERASURE - Fill the masked area with white/neutral color
        # This completely removes the original object
        white_color = [255, 255, 255]  # White background
        result_image[y_min:y_max+1, x_min:x_max+1][mask_np[y_min:y_max+1, x_min:x_max+1] > 127] = white_color
        
        print("✅ Direct erasure completed - original object removed")
        
        # Step 7: Place furniture on the erased area
        # Create a mask for the furniture placement area
        mask_roi = mask_np[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        
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
        
        print("✅ Furniture placement on erased area completed")
        return result_image
        
    except Exception as e:
        print(f"Direct erasure replacement failed: {e}")
        return original_image.copy()

def simple_overlay_replacement(original_image, mask_np, suggested_furniture, thumbnail_folder):
    """
    Simple Overlay Approach (Fallback):
    Just overlay furniture on top of the masked area without any erasure
    """
    try:
        print("Starting simple overlay replacement...")
        
        # Create a copy of the original image
        result_image = original_image.copy()
        
        # Get furniture image
        furniture_img_path = os.path.join(thumbnail_folder, suggested_furniture + ".png")
        if not os.path.exists(furniture_img_path):
            raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
        
        furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError("Could not load furniture image")
        
        # Get mask bounds
        y_indices, x_indices = np.where(mask_np > 127)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        
        # Scale and resize furniture
        furniture_h, furniture_w = furniture_image.shape[:2]
        scale = min(mask_width / furniture_w, mask_height / furniture_h)
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)
        
        # Simple overlay - just place furniture on top
        mask_roi = mask_np[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        
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
        
        print("✅ Simple overlay completed")
        return result_image
        
    except Exception as e:
        print(f"Simple overlay replacement failed: {e}")
        return original_image.copy()

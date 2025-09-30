# AI Object Detection and Selection System
# This replaces manual masking with AI-powered object detection

import cv2
import numpy as np
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import base64
from PIL import Image
from io import BytesIO

def pil_image_to_base64(image):
    """Convert PIL image to Base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def detect_objects_in_image(image_path, sam_model, mask_generator):
    """
    Use SAM (Segment Anything Model) to detect all objects in the image
    Returns a list of detected objects with their masks and bounding boxes
    """
    try:
        print("Detecting objects in image using SAM...")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks for all objects
        masks = mask_generator.generate(image_rgb)
        
        print(f"Found {len(masks)} objects in the image")
        
        # Process and filter masks
        detected_objects = []
        for i, mask_data in enumerate(masks):
            # Filter out very small objects (likely noise)
            if mask_data['area'] < 1000:  # Minimum area threshold
                continue
                
            # Get bounding box
            bbox = mask_data['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Get mask
            mask = mask_data['segmentation']
            
            # Create object info
            object_info = {
                'id': i,
                'name': f"Object {i+1}",  # We'll improve this with AI naming later
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': mask_data['area'],
                'mask': mask,
                'confidence': mask_data.get('stability_score', 0.8)
            }
            
            detected_objects.append(object_info)
        
        # Sort by area (largest objects first)
        detected_objects.sort(key=lambda x: x['area'], reverse=True)
        
        print(f"Returning {len(detected_objects)} significant objects")
        return detected_objects
        
    except Exception as e:
        print(f"Object detection failed: {e}")
        return []

def get_object_preview_images(image_path, detected_objects):
    """
    Create preview images for each detected object
    Returns base64 encoded images for the frontend
    """
    try:
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        preview_images = []
        
        for obj in detected_objects:
            # Get bounding box
            x, y, w, h = obj['bbox']
            
            # Crop the object from the original image
            cropped_object = image[y:y+h, x:x+w]
            
            # Create a preview with a colored border
            preview = cropped_object.copy()
            cv2.rectangle(preview, (0, 0), (w-1, h-1), (0, 255, 0), 3)  # Green border
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', preview)
            preview_base64 = base64.b64encode(buffer).decode('utf-8')
            
            preview_images.append({
                'object_id': obj['id'],
                'preview_image': f"data:image/jpeg;base64,{preview_base64}",
                'bbox': obj['bbox'],
                'area': obj['area']
            })
        
        return preview_images
        
    except Exception as e:
        print(f"Preview generation failed: {e}")
        return []

def replace_selected_object(image_path, selected_object_id, detected_objects, furniture_image_path):
    """
    Replace the selected object with the chosen furniture
    """
    try:
        print(f"Replacing object {selected_object_id} with furniture...")
        print(f"Detected objects count: {len(detected_objects)}")
        print(f"Looking for object ID: {selected_object_id}")
        
        # Find the selected object
        selected_object = None
        for obj in detected_objects:
            print(f"Checking object ID: {obj['id']}")
            if obj['id'] == selected_object_id:
                selected_object = obj
                print(f"Found selected object: {obj}")
                break
        
        if not selected_object:
            raise ValueError(f"Object {selected_object_id} not found")
        
        # Load the original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Load the furniture image
        furniture_image = cv2.imread(furniture_image_path, cv2.IMREAD_UNCHANGED)
        if furniture_image is None:
            raise ValueError(f"Could not load furniture image: {furniture_image_path}")
        
        # Get the object's mask
        object_mask = selected_object['mask']
        
        # Get bounding box
        x, y, w, h = selected_object['bbox']
        
        # Step 1: Remove the original object by filling with white
        # Convert mask to the same size as the image
        mask_full = np.zeros(original_image.shape[:2], dtype=np.uint8)
        mask_full[y:y+h, x:x+w] = (object_mask[y:y+h, x:x+w] * 255).astype(np.uint8)
        
        # Use inpainting to naturally remove the object
        inpaint_radius = 5
        inpainted_image = cv2.inpaint(original_image, mask_full, inpaint_radius, cv2.INPAINT_TELEA)
        
        print("✅ Original object removed using inpainting")
        
        # Step 2: Scale furniture to fit the object area
        furniture_h, furniture_w = furniture_image.shape[:2]
        scale = min(w / furniture_w, h / furniture_h)
        new_w = max(1, int(furniture_w * scale))
        new_h = max(1, int(furniture_h * scale))
        
        # Resize furniture
        furniture_resized = cv2.resize(furniture_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center furniture within the object area
        offset_x = x + max(0, (w - new_w) // 2)
        offset_y = y + max(0, (h - new_h) // 2)
        
        # Step 3: Place furniture on the inpainted area
        for fy in range(new_h):
            for fx in range(new_w):
                if (fy + offset_y < inpainted_image.shape[0] and 
                    fx + offset_x < inpainted_image.shape[1]):
                    
                    if furniture_resized.shape[2] == 4:  # Has alpha channel
                        alpha = furniture_resized[fy, fx, 3] / 255.0
                        if alpha > 0.1:  # Not transparent
                            # Blend with background
                            bg_color = inpainted_image[fy + offset_y, fx + offset_x]
                            furniture_color = furniture_resized[fy, fx, :3]
                            blended_color = alpha * furniture_color + (1 - alpha) * bg_color
                            inpainted_image[fy + offset_y, fx + offset_x] = blended_color.astype(np.uint8)
                    else:  # No alpha channel
                        inpainted_image[fy + offset_y, fx + offset_x] = furniture_resized[fy, fx, :3]
        
        print("✅ Furniture placed successfully")
        return inpainted_image
        
    except Exception as e:
        print(f"Object replacement failed: {e}")
        return None

def improve_object_names_with_ai(detected_objects, image_path):
    """
    Use AI to give better names to detected objects
    This is optional - for now we'll use generic names
    """
    # For now, return objects with generic names
    # Later we can integrate with vision models to get better names
    for i, obj in enumerate(detected_objects):
        # Simple naming based on size and position
        if obj['area'] > 50000:
            obj['name'] = "Large Furniture"
        elif obj['area'] > 20000:
            obj['name'] = "Medium Object"
        else:
            obj['name'] = "Small Object"
    
    return detected_objects

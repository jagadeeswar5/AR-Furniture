import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import openai
import logging
from dotenv import load_dotenv
import cv2
from io import BytesIO
from base64 import b64encode, b64decode
import base64
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Load environment variables from .env file
# Try backend/.env and project-root/.env for convenience
try:
    load_dotenv()
    backend_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    project_root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(backend_env):
        load_dotenv(backend_env, override=True)
    if os.path.exists(project_root_env):
        load_dotenv(project_root_env, override=True)
except Exception as _:
    pass

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Use environment variable for API key

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FURNITURE_FOLDER = os.path.join(BASE_DIR, "furniture_models/sofas")
THUMBNAIL_FOLDER = os.path.join(FURNITURE_FOLDER, "thumbnails")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Detect CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the SAM model
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"  # Update the path if needed
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

# Global variables for storing images and masks
GLOBAL_IMAGE = None
GLOBAL_MASK = None

# Load Pre-trained ResNet for Feature Extraction
resnet = models.resnet50(weights='DEFAULT')  # Updated to use 'weights' instead of 'pretrained'
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
resnet.to(device)

# Image Preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

logger.info("Models Loaded Successfully ✅")

# Sofa Details (Prices and Descriptions)
sofa_details = {
    "black_sofa": {
        "name": "Black Sofa",
        "price": 1000.00,
        "description": "A sleek and modern black sofa, perfect for contemporary spaces.",
        "style": "modern",
        "color": "black",
        "material": "leather",
        "dimensions": "84\"W x 36\"D x 32\"H"
    },
    "blue_sofa": {
        "name": "Blue Sofa",
        "price": 1200.00,
        "description": "A stylish blue sofa, ideal for adding a pop of color to your space.",
        "style": "contemporary",
        "color": "blue",
        "material": "velvet",
        "dimensions": "90\"W x 38\"D x 30\"H"
    },
    "modern_sofa2": {
        "name": "Modern Sofa 2",
        "price": 1100.00,
        "description": "A stylish and comfortable sofa, ideal for modern spaces.",
        "style": "modern",
        "color": "gray",
        "material": "linen",
        "dimensions": "96\"W x 35\"D x 31\"H"
    }
}

# ------------------------------
# Cosine-similarity recommender (text-based)
# ------------------------------
# This module adds an embeddings + cosine similarity utility without changing
# existing endpoints. Call recommend_furniture_by_cosine(user_message)
# to get a ranked list of inventory items.

# In-memory cache for inventory text embeddings
INVENTORY_EMBEDDINGS = {}

def _build_inventory_text(sofa_key: str) -> str:
    meta = sofa_details.get(sofa_key, {})
    parts = [
        meta.get("name", ""),
        meta.get("description", ""),
        meta.get("style", ""),
        meta.get("color", ""),
        meta.get("material", ""),
        meta.get("dimensions", ""),
    ]
    return " ".join(p for p in parts if p)

def _get_text_embedding(text: str):
    try:
        emb = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text.strip()[:4000],
        )
        return np.array(emb.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

def _ensure_inventory_embeddings():
    global INVENTORY_EMBEDDINGS
    if INVENTORY_EMBEDDINGS:
        return
    for filename in os.listdir(FURNITURE_FOLDER):
        if not filename.endswith('.glb'):
            continue
        key = filename.replace('.glb', '')
        if key not in sofa_details:
            continue
        text = _build_inventory_text(key)
        vec = _get_text_embedding(text)
        if vec is not None:
            INVENTORY_EMBEDDINGS[key] = vec

def recommend_furniture_by_cosine(user_message: str, top_k: int = 3):
    """Return top_k inventory items ranked by cosine similarity to user_message.

    This function is additive and does not modify existing behavior. It can be
    called by the chat or inventory flow to rank items.
    """
    try:
        _ensure_inventory_embeddings()
        query_vec = _get_text_embedding(user_message or "")
        if query_vec is None or not INVENTORY_EMBEDDINGS:
            # Fallback: simple keyword scoring if embeddings unavailable
            logger.warning("Embeddings unavailable; using keyword fallback")
            tokens = {t.lower() for t in (user_message or "").split()}
            scores = []
            for key, meta in sofa_details.items():
                text = _build_inventory_text(key).lower()
                score = sum(1 for t in tokens if t in text)
                scores.append((key, float(score)))
        else:
            scores = []
            for key, vec in INVENTORY_EMBEDDINGS.items():
                sim = float(cosine_similarity([query_vec], [vec])[0][0])
                scores.append((key, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: max(1, top_k)]
        results = []
        for key, score in top:
            meta = sofa_details.get(key, {})
            results.append({
                "key": key,
                "name": meta.get("name", key),
                "score": round(float(score), 4),
                "thumbnail": f"http://192.168.0.7:8000/thumbnails/{key}.png",
                "glb_model": f"http://192.168.0.7:8000/furniture/{key}.glb",
                **meta,
            })
        return results
    except Exception as e:
        logger.error(f"cosine recommender error: {e}")
        return []

# New endpoint to get initial furniture data
@app.get("/get-initial-furniture")
async def get_initial_furniture():
    """Endpoint to get initial furniture data when the app loads"""
    try:
        # Prepare inventory list with all available furniture
        inventory = []
        for filename in os.listdir(FURNITURE_FOLDER):
            if filename.endswith(".glb"):
                sofa_name = filename.replace(".glb", "")
                if sofa_name in sofa_details:
                    item = {
                        "filename": sofa_name,  # Add filename field for backend calls
                        "name": sofa_details[sofa_name]["name"],  # Display name
                        "thumbnail": f"http://192.168.0.7:8000/thumbnails/{sofa_name}.png",
                        "glb_model": f"http://192.168.0.7:8000/furniture/{sofa_name}.glb",  # GLB model URL
                        **sofa_details[sofa_name]
                    }
                    inventory.append(item)
        
        return JSONResponse(content={
            "sofa_details": sofa_details,
            "inventory": inventory
        })
    except Exception as e:
        logger.error(f"Error in /get-initial-furniture: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Object Detection Endpoints
@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """Use AI to detect all objects in the uploaded image"""
    try:
        logger.info("Detecting objects in uploaded image...")
        
        # Save the uploaded image
        image_data = await file.read()
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(image_data)
        
        # Import AI detection functions
        from ai_object_detection import detect_objects_in_image, get_object_preview_images, improve_object_names_with_ai
        
        # Detect objects using SAM
        detected_objects = detect_objects_in_image(image_path, sam, mask_generator)
        
        if not detected_objects:
            return JSONResponse(content={
                "message": "No objects detected in the image",
                "objects": []
            })
        
        # Improve object names with AI
        detected_objects = improve_object_names_with_ai(detected_objects, image_path)
        
        # Generate preview images
        preview_images = get_object_preview_images(image_path, detected_objects)
        
        # Prepare response
        objects_info = []
        for i, obj in enumerate(detected_objects):
            objects_info.append({
                "id": obj['id'],
                "name": obj['name'],
                "bbox": obj['bbox'],
                "area": obj['area'],
                "confidence": obj['confidence'],
                "preview": preview_images[i]['preview_image'] if i < len(preview_images) else None
            })
        
        return JSONResponse(content={
            "message": f"Found {len(detected_objects)} objects in the image",
            "objects": objects_info,
            "image_url": f"http://127.0.0.1:8000/uploads/{file.filename}"
        })
        
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/replace-object")
async def replace_object(data: dict):
    """Replace the selected object with chosen furniture"""
    try:
        image_filename = data.get("image_filename")
        object_id = data.get("object_id")
        furniture_name = data.get("furniture_name")
        
        if not all([image_filename, object_id is not None, furniture_name]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        logger.info(f"Replacing object {object_id} with {furniture_name}")
        
        # Import AI detection functions
        from ai_object_detection import detect_objects_in_image, replace_selected_object
        
        # Get image path
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get furniture path
        furniture_path = os.path.join(THUMBNAIL_FOLDER, furniture_name + ".png")
        if not os.path.exists(furniture_path):
            raise HTTPException(status_code=404, detail="Furniture image not found")
        
        # Detect objects again (we need the mask data)
        detected_objects = detect_objects_in_image(image_path, sam, mask_generator)
        
        # Replace the selected object
        logger.info(f"Calling replace_selected_object with: image_path={image_path}, object_id={object_id}, detected_objects_count={len(detected_objects)}, furniture_path={furniture_path}")
        result_image = replace_selected_object(image_path, object_id, detected_objects, furniture_path)
        logger.info(f"replace_selected_object returned: {type(result_image)}")
        
        if result_image is None:
            raise HTTPException(status_code=500, detail="Object replacement failed")
        
        # Save the result
        result_filename = f"replaced_{image_filename}"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)
        
        # Convert to base64 for frontend
        from ai_object_detection import pil_image_to_base64
        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        result_base64 = pil_image_to_base64(result_pil)
        
        return JSONResponse(content={
            "message": "Object replaced successfully!",
            "result_image": result_base64,
            "result_url": f"http://127.0.0.1:8000/uploads/{result_filename}"
        })
        
    except Exception as e:
        logger.error(f"Error in object replacement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Extract Features from an Image
def extract_features(image_path):
    try:
        logger.info(f"Extracting features from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet(image)
        return features.cpu().numpy().flatten()
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

# Find the Most Suitable Sofa with Reasoning
def find_most_suitable_sofa(room_features):
    try:
        logger.info("Finding the most suitable sofa...")
        sofa_features = {}
        for filename in os.listdir(THUMBNAIL_FOLDER):
            if filename.endswith(".png"):
                sofa_path = os.path.join(THUMBNAIL_FOLDER, filename)
                logger.info(f"Processing thumbnail: {sofa_path}")

                features = extract_features(sofa_path)
                sofa_features[filename] = features

        if not sofa_features:
            logger.info("No sofa features found.")
            return None, None, "No furniture available."

        similarities = {sofa: cosine_similarity([room_features], [features])[0][0] for sofa, features in sofa_features.items()}
        sorted_sofas = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        most_suitable_sofa = sorted_sofas[0][0]
        similarity_score = sorted_sofas[0][1]
        reason = f"This sofa matches the room's style and color scheme with a similarity score of {similarity_score:.2f}."

        logger.info(f"Most suitable sofa: {most_suitable_sofa}, Similarity score: {similarity_score}")
        return most_suitable_sofa, similarity_score, reason, sorted_sofas
    except Exception as e:
        logger.error(f"Error finding suitable sofa: {e}")
        raise

# Inpaint Sofa into the Uploaded Image using OpenCV
def inpaint_sofa_into_image(uploaded_image_path, sofa_image_path, mask_path, output_path):
    try:
        # Load original image, sofa, and mask
        uploaded_image = cv2.imread(uploaded_image_path)
        sofa_image = cv2.imread(sofa_image_path, cv2.IMREAD_UNCHANGED)  
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure mask is binary
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Step 1: Get bounding box of the mask area
        y_indices, x_indices = np.where(mask == 255)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("No valid mask found.")

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Step 2: Compute new sofa size based on mask dimensions
        mask_width = x_max - x_min
        mask_height = y_max - y_min

        # Step 3: Resize sofa to fit inside the mask area while preserving aspect ratio
        sh, sw = sofa_image.shape[:2]
        if sw == 0 or sh == 0:
            raise ValueError("Invalid sofa image dimensions.")
        scale = min(mask_width / sw, mask_height / sh)
        new_w = max(1, int(sw * scale))
        new_h = max(1, int(sh * scale))
        sofa_resized = cv2.resize(sofa_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Step 4: Create a copy of the original image to avoid modifying it directly
        final_image = uploaded_image.copy()

        # Step 4.5: Compute placement to center within mask bounds
        offset_x = x_min + max(0, (mask_width - new_w) // 2)
        offset_y = y_min + max(0, (mask_height - new_h) // 2)

        # Feather the mask edges for soft blending
        feather = 7  # odd radius
        soft_mask_full = cv2.GaussianBlur(mask, (feather, feather), 0)
        soft_mask_crop = soft_mask_full[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        soft_mask_norm = (soft_mask_crop.astype(np.float32) / 255.0)[..., None]

        # Prepare channels
        if sofa_resized.shape[2] == 4:  
            sofa_rgb = sofa_resized[:, :, :3].astype(np.float32)
            sofa_alpha = (sofa_resized[:, :, 3:4].astype(np.float32) / 255.0)
            blend_alpha = np.clip(soft_mask_norm * sofa_alpha, 0.0, 1.0)
        else:
            sofa_rgb = sofa_resized[:, :, :3].astype(np.float32)
            blend_alpha = soft_mask_norm

        # Extract target ROI
        roi = final_image[offset_y:offset_y+new_h, offset_x:offset_x+new_w].astype(np.float32)
        # Alpha blend
        blended = sofa_rgb * blend_alpha + roi * (1.0 - blend_alpha)
        final_image[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = blended.astype(np.uint8)

        # Save the final image
        cv2.imwrite(output_path, final_image)
        return output_path

    except Exception as e:
        raise ValueError(f"Error in inpainting: {e}")
    
# Upload Room Image and Suggest Suitable Furniture
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    global GLOBAL_IMAGE

    try:
        logger.info("Uploading image...")
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        GLOBAL_IMAGE = np.array(image)  # Store image globally for segmentation
        GLOBAL_MASK = None

        # Save the image to disk for feature extraction
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(image_data)

        logger.info("Image saved successfully.")

        room_features = extract_features(image_path)
        logger.info("Room features extracted.")

        best_sofa, similarity_score, reason, sorted_sofas = find_most_suitable_sofa(room_features)
        logger.info(f"Best sofa: {best_sofa}, Similarity score: {similarity_score}, Reason: {reason}")

        if not best_sofa:
            logger.info("No suitable furniture found.")
            return JSONResponse(content={"message": "No suitable furniture found."})

        sofa_name = best_sofa.replace(".png", "")
        sofa_info = sofa_details.get(sofa_name, {})

        response_data = {
            "message": "Image uploaded successfully",
            "image_url": f"http://192.168.0.7:8000/uploads/{file.filename}",
            "suggested_furniture": {
                "name": sofa_name,
                "thumbnail": f"http://192.168.0.7:8000/thumbnails/{best_sofa}",
                "glb_model": f"http://192.168.0.7:8000/furniture/{best_sofa.replace('.png', '.glb')}",
                "similarity_score": float(similarity_score),
                "reason": reason,
                "price": sofa_info.get("price", "N/A"),
                "description": sofa_info.get("description", "No description available."),
                "style": sofa_info.get("style", ""),
                "color": sofa_info.get("color", ""),
                "material": sofa_info.get("material", ""),
                "dimensions": sofa_info.get("dimensions", "")
            },
            "sorted_sofas": [(sofa, float(score)) for sofa, score in sorted_sofas],
            "sofa_details": sofa_details  # Include sofa_details in the response
        }

        logger.info(f"Returning response: {response_data}")
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error in /upload/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/segment/")
async def segment_image(data: dict):
    global GLOBAL_IMAGE, GLOBAL_MASK

    if GLOBAL_IMAGE is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    mask_base64 = data.get("mask_base64", "")

    if not mask_base64:
        raise HTTPException(status_code=400, detail="No mask provided.")

    try:
        # Decode Base64 mask
        mask_data = base64.b64decode(mask_base64.split(",")[1])  # Remove 'data:image/png;base64,' prefix
        mask_image = Image.open(BytesIO(mask_data)).convert("L")  # Convert to grayscale

        # Convert mask to binary format (thresholding)
        mask_np = np.array(mask_image)
        binary_mask = (mask_np > 128).astype(np.uint8) * 255  # Convert to binary mask

        GLOBAL_MASK = Image.fromarray(binary_mask)  # Store globally

        # SIMPLIFIED: Use only the user's manually drawn mask
        # No SAM processing - respect the user's exact mask boundaries
        logger.info("Using user's manually drawn mask (no SAM processing)")
        
        # Keep the original user mask as-is
        logger.info("✅ User mask applied successfully")
        
        # Overlay mask on original image
        segmented_img = overlay_mask_on_image(GLOBAL_IMAGE, np.array(GLOBAL_MASK))

        return JSONResponse(content={
            "mask": pil_image_to_base64(GLOBAL_MASK),
            "segmented_image": pil_image_to_base64(segmented_img),
            "message": "Segmentation successful! Transparent mask applied."
        })

    except Exception as e:
        logger.error(f"Error in segmentation: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing mask: {str(e)}")

def overlay_mask_on_image(image_array, mask_array):
    """Create a transparent mask overlay showing only the masked area."""
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array).convert("L")
    mask = mask.resize(image.size)

    # Create a white background image
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    
    # Create the segmented image by copying only the masked area
    segmented_img = white_bg.copy()
    
    # Use the mask to copy pixels from the original image
    segmented_img.paste(image, mask=mask)

    return segmented_img

# Convert PIL image to Base64
def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def encode_image(image_path):
    """Convert an image to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Inpainting API with Rectangle-Based Inpainting
@app.post("/inpainting-image")
async def inpainting(suggested_furniture: str = Form(...)):
    global GLOBAL_IMAGE, GLOBAL_MASK

    if GLOBAL_IMAGE is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    if GLOBAL_MASK is None:
        raise HTTPException(status_code=400, detail="No segmentation mask found. Please segment first.")

    try:
        logger.info("Starting inpainting process...")

        # Convert NumPy image to OpenCV format
        original_image = cv2.cvtColor(GLOBAL_IMAGE, cv2.COLOR_RGB2BGR)
        mask_np = np.array(GLOBAL_MASK.convert("L"))  # Convert mask to NumPy (grayscale)

        # Ensure mask is the same size as original image
        if mask_np.shape[:2] != original_image.shape[:2]:
            logger.info(" Resizing mask to match image size...")
            mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]))

        # Create a clean binary mask - only the drawn area should be affected
        # Threshold to ensure binary mask (255 for drawn area, 0 for background)
        mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]
        
        # No morphological operations - use exact mask boundaries
        
        # Create a copy of the original image to work with
        working_image = original_image.copy()

        # **Step 1: Extract the masked area and prepare for ChatGPT replacement**
        logger.info(" Preparing masked area for ChatGPT furniture replacement...")
        
        # Create a copy of the working image
        final_image = working_image.copy()
        
        # Get the bounding box of the mask
        mask_coords = np.where(mask_np > 0)
        if len(mask_coords[0]) == 0:
            raise HTTPException(status_code=400, detail="No valid mask area found")
            
        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
        
        mask_height = y_max - y_min + 1
        mask_width = x_max - x_min + 1
        
        # Debug: Log mask coordinates
        logger.info(f"Mask coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        logger.info(f"Mask dimensions: width={mask_width}, height={mask_height}")
        logger.info(f"Image dimensions: width={original_image.shape[1]}, height={original_image.shape[0]}")
        
        # Extract the masked area from the original image
        masked_area = working_image[y_min:y_max+1, x_min:x_max+1].copy()
        
        # Create a clean mask for the extracted area
        area_mask = mask_np[y_min:y_max+1, x_min:x_max+1]
        
        # **Step 2: Simple furniture overlay (no background processing)**
        logger.info(" Using simple furniture overlay...")
        
        try:
            # Get furniture image
            furniture_img_path = os.path.join(THUMBNAIL_FOLDER, suggested_furniture + ".png")
            if not os.path.exists(furniture_img_path):
                raise FileNotFoundError(f"Furniture image not found: {furniture_img_path}")
            
            furniture_image = cv2.imread(furniture_img_path, cv2.IMREAD_UNCHANGED)
            if furniture_image is None:
                raise ValueError("Could not load furniture image")
            
            # Resize furniture to match mask dimensions exactly
            furniture_resized = cv2.resize(furniture_image, (mask_width, mask_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Debug: Check furniture image properties
            logger.info(f"Furniture image shape: {furniture_image.shape}")
            logger.info(f"Furniture resized shape: {furniture_resized.shape}")
            if furniture_resized.shape[2] == 4:  # Has alpha channel
                # Check if there are transparent pixels at the bottom
                bottom_row = furniture_resized[-1, :, 3]  # Alpha channel of bottom row
                transparent_pixels = np.sum(bottom_row < 128)
                logger.info(f"Bottom row transparent pixels: {transparent_pixels}/{furniture_resized.shape[1]}")
                if transparent_pixels > furniture_resized.shape[1] * 0.5:
                    logger.warning("Furniture has many transparent pixels at bottom - this might cause floating!")
            
            # Create the final image starting with the original (no background processing)
            logger.info("Creating final image with original background...")
            final_image = original_image.copy()
            
            # Place furniture exactly at mask position - if mask is on floor, furniture will be on floor
            offset_x = x_min
            offset_y = y_min
            
            # If furniture has transparent bottom, adjust placement to ensure floor contact
            if furniture_resized.shape[2] == 4:  # Has alpha channel
                # Find the bottom-most non-transparent row
                alpha_channel = furniture_resized[:, :, 3]
                non_transparent_rows = np.any(alpha_channel > 128, axis=1)
                if np.any(non_transparent_rows):
                    bottom_non_transparent = np.where(non_transparent_rows)[0][-1]
                    # Adjust placement so the actual furniture bottom touches the mask bottom
                    offset_y = y_max - bottom_non_transparent
                    logger.info(f"Adjusted furniture placement to ensure floor contact: y={offset_y}")
                    logger.info(f"Furniture actual bottom row: {bottom_non_transparent}")
            
            logger.info(f"Furniture placement: x={offset_x}, y={offset_y}, size={mask_width}x{mask_height}")
            logger.info(f"Mask coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            
            # Simple furniture overlay on original image (no background processing)
            logger.info(f"Simple furniture overlay: {mask_width}x{mask_height}")
            
            # Apply furniture to every pixel - simple overlay, no background processing
            for fy in range(mask_height):
                for fx in range(mask_width):
                    if (fy + offset_y < final_image.shape[0] and 
                        fx + offset_x < final_image.shape[1] and
                        fy + offset_y >= 0 and fx + offset_x >= 0):
                        
                        # Get furniture pixel
                        furniture_pixel = furniture_resized[fy, fx]
                        
                        # Check if this pixel is within the mask area
                        mask_value = mask_np[fy + offset_y, fx + offset_x]
                        
                        if mask_value > 50:  # Only place furniture where mask is strong
                            # Place furniture naturally - use furniture's own shape
                            if furniture_resized.shape[2] == 4:  # Has alpha channel
                                alpha = furniture_pixel[3] / 255.0
                                
                                if alpha > 0.1:  # Only place furniture where it's not transparent
                                    # Get the original background pixel
                                    background_pixel = final_image[fy + offset_y, fx + offset_x]
                                    
                                    # Blend furniture with original background using alpha
                                    blended_pixel = (
                                        alpha * furniture_pixel[:3] + 
                                        (1 - alpha) * background_pixel
                                    ).astype(np.uint8)
                                    final_image[fy + offset_y, fx + offset_x] = blended_pixel
                                # If furniture pixel is transparent, keep the original background
                            else:  # No alpha channel
                                # Use the furniture's RGB color directly
                                final_image[fy + offset_y, fx + offset_x] = furniture_pixel
            
            logger.info(" ✅ Simple furniture overlay completed successfully!")
            
        except Exception as furniture_error:
            logger.error(f"Simple furniture overlay failed: {furniture_error}")
            logger.info(" Using original image as fallback...")
            # Use the original working image as fallback
            final_image = working_image.copy()

        # Convert result to PIL and Base64 for frontend
        final_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

        logger.info(" Inpainting completed successfully! Sofa replaced, background preserved.")
        return JSONResponse(content={"image": pil_image_to_base64(final_pil), "message": "Furniture replacement complete!"})

    except Exception as e:
        logger.error(f" Inpainting error: {e}")
        raise HTTPException(status_code=500, detail=f"Inpainting error: {str(e)}")
   
# Handle User Feedback
@app.post("/feedback/")
async def handle_feedback(feedback: dict):
    try:
        user_feedback = feedback.get("feedback", "").lower()
        sorted_sofas = feedback.get("sorted_sofas", [])
        uploaded_image_filename = feedback.get("uploaded_image_filename")

        if GLOBAL_MASK is None:
            raise HTTPException(status_code=400, detail="No segmentation mask found.")

        # Save the mask locally
        mask_path = os.path.join(UPLOAD_FOLDER, f"mask_{uploaded_image_filename}")
        GLOBAL_MASK.save(mask_path)  # Save as PNG file

        # Proceed with inpainting
        uploaded_image_path = os.path.join(UPLOAD_FOLDER, uploaded_image_filename)
        sofa_image_path = os.path.join(THUMBNAIL_FOLDER, sorted_sofas[0][0])
        inpainted_image_path = os.path.join(UPLOAD_FOLDER, f"inpainted_{uploaded_image_filename}")

        inpaint_sofa_into_image(uploaded_image_path, sofa_image_path, mask_path, inpainted_image_path)

        return JSONResponse(content={
            "message": "Furniture replaced successfully!",
            "inpainted_image_url": f"http://127.0.0.1:8000/uploads/inpainted_{uploaded_image_filename}"
        })

    except Exception as e:
        logger.error(f"Error in /feedback/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced AI Chatbot API
@app.post("/chat")
async def chat(message: dict):
    try:
        user_message = message.get("message", "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")

        logger.info(f"User Message: {user_message}")

        # Enhanced system prompt
        system_prompt = """
        You are an expert interior design assistant specialized in furniture recommendations. 
        Respond conversationally while providing specific furniture suggestions from our inventory.
        
        Guidelines:
        1. Be friendly and professional
        2. Keep responses short and conversational (1-2 sentences max)
        3. Don't use markdown formatting (no asterisks, bold, etc.)
        4. ONLY recommend items that exist in the provided inventory text.
        5. If the user asks for a category not in inventory (e.g., bed, chair, table), say it's not available right now and suggest the available items.
        6. For general queries like "sofa", ask what style they prefer.
        7. Only provide detailed info when specifically asked.
        8. Always suggest uploading a room photo for best recommendations.
        9. Only discuss furniture and interior design topics.
        """

        # Create furniture knowledge base
        furniture_knowledge = "\n".join(
            [f"{name}: {details['description']} | {details['style']} style | {details['color']} | "
             f"{details['material']} | {details['dimensions']} | ${details['price']}"
             for name, details in sofa_details.items()]
        )

        # Handle price queries directly
        price_keywords = ["under", "less than", "below", "budget", "cheap", "affordable"]
        if any(keyword in user_message.lower() for keyword in price_keywords):
            price = None
            for word in user_message.split():
                if word.replace('$', '').isdigit():
                    price = float(word.replace('$', ''))
                    break
            
            if price:
                matching_items = []
                for name, details in sofa_details.items():
                    if details['price'] <= price:
                        matching_items.append(details)
                
                if matching_items:
                    item_list = "\n".join(
                        f"- {item['name']} (${item['price']}): {item['description']} "
                        f"({item['color']} {item['material']})"
                        for item in matching_items
                    )
                    
                    response_msg = (
                        f"We have {len(matching_items)} options in your budget:\n\n{item_list}\n\n"
                        "Would you like to see images of any of these?"
                    )
                    return JSONResponse(content={"message": response_msg, "type": "message"})
                else:
                    return JSONResponse(content={
                        "message": f"Currently nothing under ${price}, but our most affordable is the "
                                  f"{sofa_details['modern_sofa2']['name']} at ${sofa_details['modern_sofa2']['price']}",
                        "type": "message"
                    })

        # Block queries for categories not in the current inventory
        message_lc = user_message.lower()
        available_categories = ["sofa", "couch"]
        disallowed_categories = ["bed", "chair", "table", "desk", "dining", "stool", "wardrobe", "dresser", "nightstand"]
        if any(word in message_lc for word in disallowed_categories) and not any(word in message_lc for word in available_categories):
            available_names = ", ".join([d.get("name", k) for k, d in sofa_details.items()])
            return JSONResponse(content={
                "message": f"Right now our inventory features sofas only ({available_names}). Beds and other categories aren't available yet.",
                        "type": "message"
                    })

        # Get AI response with graceful fallback when OpenAI is unavailable/quota exceeded
        try:
            # Prefer a modern lightweight model; fall back if unavailable
            model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": f"Current inventory:\n{furniture_knowledge}"},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200
            )
            chatbot_response = response.choices[0].message.content
        except Exception as openai_error:
            logger.warning(f"OpenAI chat failed (model={os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}): {openai_error}")
            chatbot_response = (
                "I'm currently offline, but I can still help with our inventory. "
                "You can ask to 'show options' or upload a room photo for recommendations."
            )
        
        # Check if we should show inventory - only for specific requests
        show_inventory = any(phrase in user_message.lower() for phrase in 
                           ["show inventory", "show options", "what do you have", "list furniture", "see inventory"])
        
        if show_inventory:
            furniture_list = []
            for filename in os.listdir(FURNITURE_FOLDER):
                if filename.endswith(".glb"):
                    sofa_name = filename.replace(".glb", "")
                    sofa_info = sofa_details.get(sofa_name, {})
                    thumbnail_filename = filename.replace(".glb", ".png")
                    
                    if os.path.exists(os.path.join(THUMBNAIL_FOLDER, thumbnail_filename)):
                        furniture_list.append({
                            "name": sofa_name,
                            "thumbnail": f"http://127.0.0.1:8000/thumbnails/{thumbnail_filename}",
                            **{k: v for k, v in sofa_info.items() if k != 'name'}
                        })

            return JSONResponse(content={
                "message": chatbot_response,
                "inventory": furniture_list,
                "type": "inventory"
            })
        
        return JSONResponse(content={
            "message": chatbot_response,
            "type": "message"
        })
        
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AR Model Endpoint
@app.get("/ar-model/{furniture_name}")
async def get_ar_model(furniture_name: str):
    """Get AR model URL for QR code generation"""
    try:
        glb_path = os.path.join(FURNITURE_FOLDER, f"{furniture_name}.glb")
        if not os.path.exists(glb_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get local IP for mobile access
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # Create AR-compatible URL
        model_url = f"http://192.168.0.7:8000/furniture/{furniture_name}.glb"
        
        return JSONResponse(content={
            "model_url": model_url,
            "ar_url": f"https://arvr.google.com/scene-viewer/1.0?file={model_url}&mode=ar_only",
            "furniture_name": furniture_name
        })
        
    except Exception as e:
        logger.error(f"Error getting AR model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve Static Files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAIL_FOLDER), name="thumbnails")
app.mount("/furniture", StaticFiles(directory=FURNITURE_FOLDER), name="furniture")

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
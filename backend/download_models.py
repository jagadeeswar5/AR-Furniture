import os
from diffusers import StableDiffusionInpaintPipeline

# Define model save path
model_path = os.path.join("..", "models", "sd-v1-5-inpainting")

# Create models directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

# Download the model
print("🔄 Downloading Stable Diffusion Inpainting model...")
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
)

# Save the model locally
pipeline.save_pretrained(model_path)

# Verify that model files exist
if os.path.exists(os.path.join(model_path, "model_index.json")):
    print(f"✅ Model successfully downloaded and saved at: {model_path}")
else:
    print("❌ Model download failed. Please check your internet connection.")


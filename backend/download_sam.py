import os
import requests
from tqdm import tqdm

# Define model URL and save path
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
model_path = os.path.join("..", "models", "sam_vit_h_4b8939.pth")

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Function to download a file with progress bar
def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    t = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading SAM Model")

    with open(destination, "wb") as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

# Download the SAM model
if not os.path.exists(model_path):
    print("🔄 Downloading SAM model (sam_vit_h_4b8939.pth)...")
    download_file(sam_url, model_path)

    # Verify that model file exists
    if os.path.exists(model_path):
        print(f"✅ SAM model successfully downloaded and saved at: {model_path}")
    else:
        print("❌ SAM model download failed. Please check your internet connection.")
else:
    print("✅ SAM model already exists. No download needed.")

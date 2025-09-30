import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def process_inpainting(device, org_img, mask_img, prompt, guidance_scale=7.3, num_samples=1, seed=0):
    print("Loading Stable Diffusion Model for Inpainting...")
    model_path = "runwayml/stable-diffusion-inpainting"

    # Resize images to 512x512 for Stable Diffusion
    desired_shape = (512, 512)
    org_img = org_img.resize(desired_shape, Image.LANCZOS)
    mask_img = mask_img.resize(desired_shape, Image.LANCZOS)

    # Load the inpainting pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    print("Generating Inpainted Image...")

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)

    # Perform inpainting
    images = pipe(
        prompt=prompt,
        image=org_img,
        mask_image=mask_img,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    return images
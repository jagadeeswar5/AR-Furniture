import os
import trimesh
import pyrender
import numpy as np
from PIL import Image

# Define paths
FURNITURE_FOLDER = os.path.join("furniture_models", "sofas")  # Adjusted path
THUMBNAIL_FOLDER = os.path.join(FURNITURE_FOLDER, "thumbnails")

# Create thumbnails directory if not exists
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

def generate_thumbnail(glb_file):
    """Generates a thumbnail for a given GLB file."""
    try:
        print(f"Processing {glb_file}...")

        # Load the GLB model
        scene = trimesh.load(glb_file)

        # Extract the main mesh if the file is a scene
        if isinstance(scene, trimesh.Scene):
            if not scene.geometry:
                raise ValueError("No geometry found in the scene.")

            # Merge all meshes into a single mesh
            mesh = trimesh.util.concatenate(scene.geometry.values())
        else:
            mesh = scene  # If it's already a mesh, use it directly

        # Create a pyrender scene
        render_scene = pyrender.Scene()
        render_scene.add(pyrender.Mesh.from_trimesh(mesh))

        # **FIX:** Check if the scene has a camera and add one if missing
        if not any(isinstance(node, pyrender.Camera) for node in render_scene.nodes):
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera_pose = np.eye(4)
            camera_pose[2, 3] = 2  # Move camera back
            render_scene.add(camera, pose=camera_pose)

        # Set up lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        render_scene.add(light)

        # Render the scene
        viewer = pyrender.OffscreenRenderer(300, 300)  # Adjust resolution as needed
        color, _ = viewer.render(render_scene)
        viewer.delete()

        # Convert to PIL image and save
        img = Image.fromarray(color)
        thumbnail_path = os.path.join(THUMBNAIL_FOLDER, os.path.basename(glb_file).replace(".glb", ".jpg"))
        img.save(thumbnail_path)

        print(f"✅ Thumbnail saved: {thumbnail_path}")
    except Exception as e:
        print(f"❌ Error processing {glb_file}: {e}")

# Process all `.glb` files in the furniture folder
for file in os.listdir(FURNITURE_FOLDER):
    if file.endswith(".glb"):
        glb_path = os.path.join(FURNITURE_FOLDER, file)
        generate_thumbnail(glb_path)

print("\n🎉 Thumbnails Generated Successfully for all GLB models! ✅")

## AR Furniture App – Demo Guide and Code Walkthrough

This guide is designed for a professor demo. It includes: what to show, how to explain it, where the code lives, and talking points for each function/endpoint. Use it as a script.

### 1) One‑minute overview (say this up front)
- The app lets users upload a photo of a room, draw a mask over the area to replace, get AI recommendations, and preview furniture in place and in AR on mobile.
- Stack: FastAPI (Python) backend with OpenCV/Pillow for segmentation/inpainting/overlay; frontend is HTML/CSS/JS with Fabric.js for mask drawing, Model Viewer for 3D preview, and QR codes for AR.
- Current scope delivers the core loop: recommend → segment → replace → AR preview.

### 2) Live demo flow (follow these steps during the demo)
1. Open `frontend/home.html` → click "Try the App".
2. In the chatbot, type a simple query (e.g., “I need a black sofa”).
3. Upload a room photo when prompted.
4. Draw a rectangle over the existing furniture area and click Confirm. Show the segmented preview.
5. Click Yes on a suggested item. The backend replaces the masked region with the selected furniture; show the result image.
6. Adjust with the Opacity slider and Enlarge/Shrink.
7. Click “Generate QR Code for Mobile AR” → scan on Android (or use the iOS fallback link).

Tips
- Call out that confirming the mask only segments; inpainting/replacement happens only after choosing furniture.
- If QR generation fails, the UI prints platform‑specific fallback links.

### 3) Architecture at a glance (whiteboard or say this)
- Frontend (`frontend/index.html`, `styles.css`, `home.html`): UI, Fabric.js mask, QR code, Model Viewer.
- Backend (`backend/main.py`): FastAPI endpoints for chat, segmentation, inpainting/overlay, and static serving of thumbnails/models.
- Image processing: OpenCV + Pillow. 3D models (`backend/furniture_models/*`).

### 4) Frontend walkthrough – files and key functions
- `frontend/home.html`
  - Landing hero with background image and floating emoji motifs.
  - CTA to `index.html`.
- `frontend/index.html`
  - Chat UI: sends messages to `/chat`.
  - Masking: Fabric.js canvas; the user draws a rectangle, which we export as a binary mask.
  - Key functions to mention:
    - `confirmMask()`: posts the image + drawn mask to backend `/segment/`. Stores `window.currentMask` for later.
    - `handleUserResponse(response)`: if "yes" and a mask exists → calls `inpaintImage()`; else asks to draw a mask.
    - `inpaintImage()`: posts selected furniture + current mask to `/inpainting-image` and displays result.
    - `addChatbotMessage(msg)`: appends messages to chat.
    - `displayFurnitureInventory(items)`: renders thumbnails and attaches click handlers to select furniture.
    - `showQRCode()`: builds AR URL and renders a QR code; logs/fallbacks for iOS.
  - Controls: `updateOpacity()`, `resizeSofa(scale)` for UI adjustments.
- `frontend/styles.css`
  - Layout system, consistent spacing variables, responsive grid for inventory.
  - Important rule: `#segmentedImage, #inpaintedImage { object-fit: contain; max-height set }` to avoid cropping.

### 5) Backend walkthrough – endpoints and core logic
- File: `backend/main.py`
- Endpoints to explain:
  - `POST /chat` – system prompt tuned for concise, non‑markdown replies; conditional inventory display.
  - `POST /segment/` – accepts the white‑on‑black mask from the client, resizes to image size, stores as `GLOBAL_MASK`, and returns a segmented preview.
    - Talking point: we use an exact, non‑expanded binary mask for clean edges (no dilate/blur by default).
    - Helper: `overlay_mask_on_image(image_array, mask_array)` – creates a transparent mask overlay showing only masked pixels against white.
  - `POST /inpainting-image` – the main replacement pipeline:
    - Validates `GLOBAL_IMAGE` and `GLOBAL_MASK`.
    - Computes bounding box of the mask and resizes the selected furniture to match (`mask_width`, `mask_height`).
    - Placement: computes offsets (`x_min`, `y_min`) and adjusts bottom alignment if the furniture image contains transparent bottom rows.
    - Pixel loop applies alpha blending (or direct RGB for non‑alpha art) onto a copy of the original image.
    - Returns base64 PNG to the frontend.
  - Static: serving thumbnails from `backend/furniture_models/sofas/thumbnails` and `.glb` models.

Key images/variables
- `GLOBAL_IMAGE`: last uploaded room image (NumPy RGB)
- `GLOBAL_MASK`: last confirmed binary mask (PIL L)
- `THUMBNAIL_FOLDER`: path to PNG thumbnails used in UI

### 6) Data & assets
- `backend/furniture_models/sofas/*.glb` – example models.
- Thumbnails are generated and served for the inventory.
- No external model repository integration in this version (proposal item pending).

### 7) What to emphasize (learning/outcomes)
- Robust mask → exact placement: The switch from heavy morphology to exact masks removed artifacts.
- Separation of concerns: “Confirm Mask” only segments; “Yes” triggers replacement.
- AR pragmatism: Android via Scene Viewer; iOS shows HTTPS fallback link due to platform requirements.
- Iterative engineering: trade‑offs between inpainting vs. direct overlay, and how UX guided the final approach.

### 8) Known limitations (acknowledge proactively)
- No native AR room scan (ARKit/ARCore) or Unity pipeline yet.
- No large‑scale dataset ingestion/training; segmentation is mask‑driven, not learned per scene.
- No external model API integration (e.g., Sketchfab) yet.
- Optimal placement is heuristic (mask bounding box + bottom alignment), not an ML optimizer.

### 9) Suggested Q&A cheat sheet
- Q: Why do some artifacts appear around edges?
  - A: When user masks are imprecise, edges can show. We chose exact masks to avoid over‑expansion. Alpha blending reduces hard seams.
- Q: Why Android AR but not full iOS?
  - A: iOS enforces HTTPS/Quick Look. We provide a working HTTPS fallback link; native pipeline is planned.
- Q: Can the system auto‑remove the original furniture?
  - A: We implemented both inpainting and direct overlay; current default favors clean overlay with alpha for predictability.

### 10) How to run for the demo
- Terminal 1 (backend):
  - `cd backend`
  - Activate venv if needed; `python main.py`
- Terminal 2 (frontend):
  - `cd frontend`
  - `python -m http.server 3000`
- Open `http://localhost:3000/home.html`.

### 11) Timeboxed script (5–8 minutes)
1. 0:00–0:45 – Vision and problem statement.
2. 0:45–1:30 – Architecture slide (this section) + tech stack.
3. 1:30–3:30 – Live demo (upload → mask → confirm → replace → adjust → AR).
4. 3:30–4:30 – Code tour (endpoints and key functions as listed).
5. 4:30–5:30 – Results and limitations; what’s next.
6. 5:30–8:00 – Q&A.

### 12) What’s next (if asked about roadmap)
- Native AR scan and layout reconstruction (ARKit/ARCore), Unity visualization.
- External model integration (Sketchfab) + style consistency engine.
- Learned placement (RL/search‑based) and scene constraints.
- Cloud deployment + CI/CD; formal user studies.

— End of guide —



# Smoke Tests

## Core flow
1. Open `home.html` → click "Try the App".
2. Ask chatbot for a sofa; ensure it responds and does not mention non‑inventory categories.
3. Upload a room photo.
4. Draw mask → Confirm → segmented image appears (no cropping).
5. Recommended sofa shows with price, description, and Match score.
6. Click Yes → inpainted image appears; AR viewer shows `.glb`.
7. Generate QR → Android opens Scene Viewer; iOS link fallback visible.

Expected:
- No console errors; UI sections display as described.


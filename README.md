# 🏠 AR Furniture App

A development project exploring AI-powered furniture placement using augmented reality and computer vision.

## 📋 Table of Contents

- [Overview]
- [Features]
- [Technologies Used]
- [Installation & Setup]
- [How to Use]
- [Project Structure]
- [API Endpoints]
- [Development Notes]
- [Testing]
- [Contributing]

## 🎯 Overview

The AR Furniture App is an innovative web application that combines artificial intelligence, computer vision, and augmented reality to help users visualize furniture in their spaces. Users can upload room photos, get AI-powered furniture recommendations, and experience their chosen furniture in AR.

## ✨ Features

### 🤖 AI Chatbot Assistant
- **Technology**: OpenAI GPT integration
- **Purpose**: Provides intelligent furniture recommendations based on user queries
- **Features**:
  - Conversational interface
  - Context-aware responses
  - Inventory management
  - Style and budget recommendations

### 🎨 Image Segmentation & Inpainting
- **Technology**: 
  - Segment Anything Model (SAM) for object detection
  - OpenCV for image processing
  - Custom inpainting algorithms
- **Purpose**: Removes existing furniture and places new furniture seamlessly
- **Features**:
  - Interactive mask drawing
  - Precise object segmentation
  - Background restoration
  - Furniture overlay with alpha blending

### 📱 Augmented Reality (AR)
- **Technology**: 
  - Google Scene Viewer
  - Model Viewer Web Components
  - QR Code generation
- **Purpose**: Allows users to view 3D furniture models in AR
- **Features**:
  - Cross-platform AR support (Android)
  - QR code generation for mobile access
  - 3D model rendering
  - AR placement and interaction

### 🛋️ Furniture Management
- **Technology**: FastAPI backend with file system storage
- **Purpose**: Manages furniture inventory and 3D models
- **Features**:
  - Furniture catalog with thumbnails
  - 3D model storage (.glb files)
  - Price and description management
  - Dynamic inventory display

## 🛠️ Technologies Used

### Backend
- **Python 3.12** - Main programming language
- **FastAPI** - Web framework for API development
- **OpenCV (cv2)** - Computer vision and image processing
- **PIL (Pillow)** - Image manipulation and processing
- **NumPy** - Numerical computing
- **Segment Anything Model (SAM)** - AI object segmentation
- **OpenAI API** - AI chatbot integration
- **Uvicorn** - ASGI server

### Frontend
- **HTML5** - Structure and markup
- **CSS3** - Styling and responsive design
- **JavaScript (ES6+)** - Client-side functionality
- **Fabric.js** - Canvas manipulation for mask drawing
- **Model Viewer** - 3D model rendering
- **QRCode.js** - QR code generation
- **Fetch API** - HTTP requests

### AI & Computer Vision
- **Segment Anything Model (SAM)** - Meta's state-of-the-art segmentation model
- **OpenCV Inpainting** - Background restoration algorithms
- **Custom Alpha Blending** - Furniture overlay techniques
- **Morphological Operations** - Image processing and enhancement
 - **Cosine Similarity Recommender (text embeddings)** - Ranks inventory against user query

### AR & 3D
- **Google Scene Viewer** - Cross-platform AR viewing
- **GLB/GLTF** - 3D model format
- **WebXR** - Web-based AR/VR standards
- **Model Viewer Web Components** - 3D model display

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js (for frontend development)
- Git

### Backend Setup
```bash
# Clone the repository
git clone <repository-url>
cd AR-Furniture-App

# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv env

# Activate virtual environment
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download AI models
python download_models.py
python download_sam.py

# Start the backend server
python main.py
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Serve the frontend (using Python's built-in server)
python -m http.server 3000

# Or use any static file server
# Example with Node.js:
npx serve .
```

### Environment Variables
Create a `.env` file in the backend directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 📖 How to Use

### 1. Getting Started
1. Open your browser and navigate to `http://localhost:3000/home.html`
2. Click "🚀 Try the App" to access the main application
3. The app will load with the AI chatbot ready to assist

### 2. Chat with AI Assistant
1. Type your furniture queries in the chat input
2. Examples:
   - "I need a sofa for my living room"
   - "Show me furniture under $1000"
   - "What style goes with modern decor?"
3. The AI will provide recommendations and show available inventory

### 3. Upload Room Photo
1. When prompted, upload a photo of your room
2. The image will be displayed with drawing tools
3. Use the rectangle tool to draw a mask around furniture you want to replace

### 4. Segment and Replace Furniture
1. Click "Confirm" to segment the masked area
2. Review the segmented result
3. Select furniture from recommendations
4. Click "Yes" to apply the furniture to your room
5. Adjust opacity and size using the controls

### 5. View in AR
1. After furniture placement, click "📱 Generate QR Code for Mobile AR"
2. Scan the QR code with your mobile device
3. View the 3D furniture model in AR
4. Place and interact with the furniture in your real space

### 6. Adjust Furniture
- **Opacity**: Use the slider to adjust transparency
- **Size**: Use "🔍 Enlarge" and "🔍 Shrink" buttons
- **Position**: Furniture is automatically placed based on your mask

## 📁 Project Structure

```
AR-Furniture-App/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── ai_object_detection.py  # AI segmentation logic
│   ├── inpainting.py          # Image inpainting algorithms
│   ├── download_models.py     # Model download script
│   ├── download_sam.py        # SAM model download
│   ├── models/                # AI model storage
│   ├── furniture_models/      # 3D furniture models
│   ├── uploads/               # User uploaded images
│   └── processed/             # Processed images
├── frontend/
│   ├── index.html             # Main application page
│   ├── home.html              # Landing page
│   ├── styles.css             # Styling
│   ├── scripts.js             # Client-side JavaScript
│   └── ar-viewer.html         # AR viewing page
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## 🔌 API Endpoints

### Core Endpoints
- `POST /upload` - Upload room image
- `POST /segment/` - Segment objects in image
- `POST /inpainting-image` - Apply furniture replacement
- `POST /chat` - AI chatbot interaction
- `GET /furniture/{filename}` - Serve 3D furniture models
- `GET /thumbnails/{filename}` - Serve furniture thumbnails

### Example API Usage
```javascript
// Upload image
const formData = new FormData();
formData.append('image', imageFile);
const response = await fetch('/upload', {
    method: 'POST',
    body: formData
});

// Chat with AI
const chatResponse = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'I need a sofa' })
});
```

## 🔧 Development Notes

### Key Algorithms
1. **Image Segmentation**: Uses SAM model for precise object detection
2. **Inpainting**: Combines OpenCV algorithms with custom blending
3. **Furniture Placement**: Alpha blending with mask-based positioning
4. **AR Integration**: QR code generation with platform-specific URLs

## 🧪 Testing

All test cases and manual QA scripts live in the `test-cases/` folder at the repository root. It includes:

- `test-cases/SMOKE.md` – quick smoke flows (upload → mask → confirm → replace → AR).
- `test-cases/CHATBOT.md` – inventory-aware chatbot scenarios (e.g., disallow non‑sofa categories).
- `test-cases/RECOMMENDER.md` – cosine similarity ranking checks and fallback behavior.
- `test-cases/FRONTEND.md` – UI checks (segmented/inpainted images not cropped, score rendering, cycling alternatives).

How to execute
- Follow the steps in each markdown file; most tests are manual acceptance checks suitable for demos.
- Where API calls are shown, you can use curl or Postman.

### Performance Considerations
- Images are processed server-side for better performance
- 3D models are optimized for web delivery
- Caching implemented for frequently accessed resources
- Responsive design for mobile compatibility

### Browser Compatibility
- **Chrome/Edge**: Full support including AR features
- **Firefox**: Full support, limited AR features
- **Safari**: Full support, AR via Scene Viewer
- **Mobile**: iOS and Android AR support

## 🤝 Contributing

This is a development project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use semantic HTML and accessible CSS
- Test on multiple browsers and devices
- Document new features and APIs

## 📝 License

This project is for educational and development purposes.

## 👨‍💻 Author

**Jagadeeswar Kamireddy**
- Development project for exploring AI and AR in interior design
- Created in 2025

## 🚧 Status

This is a **development project** - Features may not work perfectly yet. The application is actively being developed and improved.

---

*For questions or support, please refer to the development documentation or create an issue in the repository.*

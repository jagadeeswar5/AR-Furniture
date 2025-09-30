const backendUrl = "http://127.0.0.1:8000"; // Update this if backend runs on another IP

let uploadedImageUrl = "";
let selectedFurniture = "";
let sortedSofas = [];
let isDrawing = false;
const maskCanvas = document.getElementById("mask-canvas");
const maskCtx = maskCanvas.getContext("2d");

// Function to display chatbot messages
function addChatbotMessage(message) {
    const chatbox = document.getElementById("chatbox");
    const newMessage = document.createElement("p");
    newMessage.innerHTML = `<strong>Chatbot:</strong> ${message}`;
    chatbox.appendChild(newMessage);
    chatbox.scrollTop = chatbox.scrollHeight;
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

// Upload Image and Get Suggested Furniture
async function uploadImage() {
    const fileInput = document.getElementById("imageUpload");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${backendUrl}/upload/`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorResponse = await response.json();
            console.error("Backend error:", errorResponse);
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        uploadedImageUrl = result.image_url;

        // Display the uploaded image
        const uploadedImage = document.getElementById("uploaded-image");
        uploadedImage.src = uploadedImageUrl;
        uploadedImage.style.display = "block";

        // Set up the mask canvas
        maskCanvas.width = uploadedImage.width;
        maskCanvas.height = uploadedImage.height;
        maskCanvas.style.display = "block";

        // Enable drawing on the mask canvas
        enableMaskDrawing();

        // Display suggested furniture
        if (result.suggested_furniture) {
            document.getElementById("suggestedFurnitureName").innerText = `Best Match: ${result.suggested_furniture.name}`;
            document.getElementById("suggestedFurnitureImage").src = result.suggested_furniture.thumbnail;
            document.getElementById("suggestedFurniturePrice").innerText = `Price: $${result.suggested_furniture.price}`;
            document.getElementById("suggestedFurnitureDescription").innerText = `Description: ${result.suggested_furniture.description}`;
            document.getElementById("suggested-furniture-section").style.display = "block";

            addChatbotMessage(`Chatbot: We recommend ${result.suggested_furniture.name}. ${result.suggested_furniture.reason}`);
            addChatbotMessage("Chatbot: Is this okay? (Click 'Yes' or 'No')");

            sortedSofas = result.sorted_sofas;
        } else {
            addChatbotMessage("Chatbot: No suitable furniture found.");
        }

    } catch (error) {
        console.error("Upload failed:", error);
        alert("Image upload failed. Check the console for details.");
    }
}

// Enable drawing on the mask canvas
function enableMaskDrawing() {
    let isDrawing = false;

    maskCanvas.addEventListener("mousedown", (e) => {
        isDrawing = true;
        maskCtx.beginPath();
        maskCtx.moveTo(e.offsetX, e.offsetY);
    });

    maskCanvas.addEventListener("mousemove", (e) => {
        if (isDrawing) {
            maskCtx.lineTo(e.offsetX, e.offsetY);
            maskCtx.strokeStyle = "white";
            maskCtx.lineWidth = 20;  // Adjust brush size
            maskCtx.stroke();
        }
    });

    maskCanvas.addEventListener("mouseup", () => {
        isDrawing = false;
    });

    maskCanvas.addEventListener("mouseleave", () => {
        isDrawing = false;
    });
}

// Handle User Feedback (Yes/No)
async function handleUserResponse(feedback) {
    if (feedback === "yes") {
        try {
            // Get the mask data as a base64 image
            const maskData = maskCanvas.toDataURL();

            const response = await fetch(`${backendUrl}/feedback/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    feedback: "yes",
                    sorted_sofas: sortedSofas,
                    uploaded_image_filename: uploadedImageUrl.split("/").pop(),
                    mask_data: maskData
                })
            });

            const result = await response.json();

            if (result.sofa_details) {
                // Show AR viewer
                document.getElementById("ar-model").src = result.sofa_details.glb_model;
                document.getElementById("ar-viewer").style.display = "block";

                // Display inpainted image
                document.getElementById("inpaintedImage").src = result.inpainted_image_url;
                document.getElementById("inpainted-image-section").style.display = "block";

                addChatbotMessage(`Chatbot: ${result.message}`);
            } else {
                addChatbotMessage("Chatbot: Sorry, details for this sofa are not available.");
            }
        } catch (error) {
            console.error("Error fetching sofa details:", error);
            addChatbotMessage("Chatbot: Something went wrong, please try again.");
        }
    } else {
        await requestAlternativeFurniture();
    }
}

// Request Next Best Sofa
async function requestAlternativeFurniture() {
    try {
        const response = await fetch(`${backendUrl}/feedback/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ feedback: "no", sorted_sofas: sortedSofas })
        });

        const result = await response.json();

        if (result.suggested_furniture) {
            document.getElementById("suggestedFurnitureName").innerText = `Next Best: ${result.suggested_furniture.name}`;
            document.getElementById("suggestedFurnitureImage").src = result.suggested_furniture.thumbnail;
            document.getElementById("suggestedFurniturePrice").innerText = `Price: $${result.suggested_furniture.price}`;
            document.getElementById("suggestedFurnitureDescription").innerText = `Description: ${result.suggested_furniture.description}`;
            addChatbotMessage(`Chatbot: We recommend ${result.suggested_furniture.name}. ${result.suggested_furniture.reason}`);
            addChatbotMessage("Chatbot: Is this okay? (Click 'Yes' or 'No')");

            sortedSofas = result.sorted_sofas;
        } else {
            addChatbotMessage("Chatbot: Sorry, we will have more inventory really soon.");
        }
    } catch (error) {
        console.error("Error fetching alternative furniture:", error);
        addChatbotMessage("Chatbot: Something went wrong, please try again.");
    }
}

// Handle user messages
async function sendMessage() {
    const userInput = document.getElementById("userInput");
    const message = userInput.value.trim();

    if (!message) {
        alert("Please enter a message.");
        return;
    }

    // Display user's message in the chatbox
    addChatbotMessage(`You: ${message}`);

    try {
        const response = await fetch(`${backendUrl}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Response from backend:", result);  // Debug logging

        // Display chatbot's response
        addChatbotMessage(`Chatbot: ${result.message}`);

        // If the chatbot returns inventory, display images
        if (result.inventory) {
            addChatbotMessage("Here is the furniture inventory:");
            displayInventory(result.inventory);
        }
    } catch (error) {
        console.error("Error sending message:", error);
        addChatbotMessage("Chatbot: Sorry, something went wrong. Please try again.");
    } finally {
        // Clear the input field
        userInput.value = "";
    }
}

// Function to display furniture inventory images
function displayInventory(inventory) {
    const inventoryContainer = document.getElementById("inventory-container");
    inventoryContainer.innerHTML = ""; // Clear previous inventory

    inventory.forEach(item => {
        if (item.thumbnail) {
            const imgElement = document.createElement("img");
            imgElement.src = item.thumbnail;
            imgElement.alt = item.name;
            imgElement.classList.add("thumbnail");

            // Select furniture on click
            imgElement.onclick = () => {
                selectedFurniture = item.name;
                addChatbotMessage(`You selected: ${item.name}`);
                addChatbotMessage(`Chatbot: ${item.name} costs $${item.price}. ${item.description}`);
            };

            inventoryContainer.appendChild(imgElement);
        }
    });
}

// Update opacity of the inpainted image
function updateOpacity() {
    const opacity = document.getElementById("opacity").value;
    document.getElementById("inpaintedImage").style.opacity = opacity;
}

// Move the inpainted sofa
function moveSofa(deltaX, deltaY) {
    const inpaintedImage = document.getElementById("inpaintedImage");
    const currentLeft = parseFloat(inpaintedImage.style.left) || 0;
    const currentTop = parseFloat(inpaintedImage.style.top) || 0;
    inpaintedImage.style.left = `${currentLeft + deltaX}px`;
    inpaintedImage.style.top = `${currentTop + deltaY}px`;
}

// Resize the inpainted sofa
function resizeSofa(scale) {
    const inpaintedImage = document.getElementById("inpaintedImage");
    const currentWidth = parseFloat(inpaintedImage.style.width) || inpaintedImage.width;
    const currentHeight = parseFloat(inpaintedImage.style.height) || inpaintedImage.height;
    inpaintedImage.style.width = `${currentWidth * scale}px`;
    inpaintedImage.style.height = `${currentHeight * scale}px`;
}

// Initialize Chatbot
addChatbotMessage("Hello! How can I help you today?");
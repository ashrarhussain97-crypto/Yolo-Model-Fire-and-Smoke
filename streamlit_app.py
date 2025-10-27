import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

# Page configuration
st.set_page_config(
    page_title="Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide"
)

# Title and description
st.title("🔥 Fire & Smoke Detection System")
st.markdown("Upload an image to detect fire or smoke using AI")

# Define your model architecture (IMPORTANT: This must match your trained model!)
# You need to replace this with YOUR actual model architecture
class FireDetectionModel(nn.Module):
    def __init__(self):
        super(FireDetectionModel, self).__init__()
        # Example architecture - REPLACE with your actual model structure!
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: fire/no-fire
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load model with caching
@st.cache_resource
def load_model():
    try:
        # Initialize model
        model = FireDetectionModel()
        
        # Load saved weights
        model.load_state_dict(torch.load('models/fire_detection_model.pt', map_location=torch.device('cpu')))
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'fire_detection_model.pt' is in the 'models/' folder")
        return None

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Load the model
model = load_model()

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This application uses deep learning to detect fire and smoke in images. "
        "Upload an image and the AI will analyze it for potential fire hazards."
    )
    st.header("How to Use")
    st.markdown("""
    1. Click 'Browse files' to upload an image
    2. Supported formats: JPG, JPEG, PNG
    3. Wait for the AI to analyze
    4. View the prediction results
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image to detect fire or smoke"
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if model is not None:
            # Preprocess the image
            with st.spinner('Analyzing image...'):
                # Preprocess
                img_tensor = preprocess_image(image)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    confidence = confidence.item()
                    predicted_class = predicted.item()
                
                # Display results
                if predicted_class == 1:  # Assuming 1 = fire detected
                    st.error("🔥 **FIRE/SMOKE DETECTED!**")
                    st.metric("Confidence", f"{confidence * 100:.2f}%")
                    st.warning("⚠️ Potential fire hazard detected in the image!")
                else:
                    st.success("✅ **NO FIRE/SMOKE DETECTED**")
                    st.metric("Confidence", f"{confidence * 100:.2f}%")
                    st.info("The image appears to be safe.")
                
                # Show progress bar
                st.progress(confidence)
        else:
            st.error("⚠️ Model not loaded. Cannot make predictions.")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload an image to begin detection")
    
    # Example section
    st.markdown("---")
    st.subheader("📊 What This App Does")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔥 Fire Detection")
        st.write("Identifies flames and fire in images")
    
    with col2:
        st.markdown("### 💨 Smoke Detection")
        st.write("Detects smoke patterns and haze")
    
    with col3:
        st.markdown("### 🎯 High Accuracy")
        st.write("AI-powered deep learning model")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Fire & Smoke Detection System | Powered by PyTorch & Streamlit"
    "</div>",
    unsafe_allow_html=True
)

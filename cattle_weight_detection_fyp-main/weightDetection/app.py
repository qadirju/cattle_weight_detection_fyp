import streamlit as st
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False

st.title("🐄 Cattle Weight Predictor")

try:
    from utils.preprocess import transform
    from utils.predict import predict_weight
except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

def load_model_on_demand():
    """Load model only when needed (lazy loading)"""
    if st.session_state.model_loaded:
        return st.session_state.model
    
    try:
        from utils.model import load_model
        st.session_state.model = load_model()
        st.session_state.model_loaded = True
        return st.session_state.model
    except Exception as e:
        st.error(f"❌ Model Loading Error: {str(e)}")
        st.info("Please ensure 'best_model.pth' exists in the same directory as app.py")
        st.stop()

st.subheader("Upload an image to predict cattle weight")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.info("Loading model and making prediction...")
        
        try:
            # Load model on demand
            model = load_model_on_demand()
            
            # Preprocess image
            img_tensor = transform(image).unsqueeze(0)
            
            # Make prediction
            weight = predict_weight(model, img_tensor)
            
            st.success(f"✓ Prediction Complete!")
            st.metric("Estimated Weight", f"{weight:.2f} kg")
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.info("Make sure the model file exists and config.py has correct MEAN_WEIGHT and STD_WEIGHT values")

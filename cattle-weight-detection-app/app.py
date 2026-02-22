import streamlit as st
from PIL import Image
from predictor import CattleWeightPredictor

st.set_page_config(page_title="Cattle Weight Estimator", layout="centered")
st.title("🐄 Cattle Weight Estimation System")
st.write("Upload a cattle image and the AI will estimate its weight.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.write("This app uses deep learning models trained in Google Colab to estimate cattle weight from images.")
st.sidebar.write("Models: Segmentation & Regression (Keras)")
st.sidebar.write("Author: FYP Team")

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    return CattleWeightPredictor()

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"Image loading failed: {e}")
        image = None

    predictor = load_model()

    if image:
        with st.spinner("Analyzing cattle..."):
            weight, confidence = predictor.predict(image)

        if weight is None or confidence is None:
            st.error("Prediction failed. Please check your model files or image format.")
        else:
            st.success(f"Estimated Weight: {weight:.1f} kg")
            st.info(f"Cattle Detection Confidence: {confidence:.1f}%")
    else:
        st.warning("Please upload a valid image.")
import streamlit as st
from PIL import Image
from predictor import CattleWeightPredictor

st.set_page_config(page_title="Cattle Weight Estimator", layout="centered")
st.title("🐄 Cattle Weight Estimation System")
st.write("Upload a cattle image and the AI will estimate its weight.")

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    return CattleWeightPredictor()

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    predictor = load_model()

    with st.spinner("Analyzing cattle..."):
        weight, confidence = predictor.predict(image)

    st.success(f"Estimated Weight: {weight:.1f} kg")
    st.info(f"Cattle Detection Confidence: {confidence:.1f}%")
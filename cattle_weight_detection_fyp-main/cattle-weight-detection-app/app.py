import streamlit as st
from PIL import Image
from predictor import CattleWeightPredictor
import traceback

# ---------------- Page Config ----------------
st.set_page_config(page_title="Cattle Weight Estimator", layout="centered")

st.title("🐄 Cattle Weight Estimation System")
st.write("Upload a cattle image and the AI will estimate its weight.")

# ---------------- Sidebar ----------------
st.sidebar.header("About")
st.sidebar.write(
    "This app uses deep learning models (Segmentation + Regression) "
    "to estimate cattle weight from images."
)
st.sidebar.write("Framework: TensorFlow / Keras")
st.sidebar.write("Project: Final Year Project")

# ---------------- Load Model (ONLY ONCE) ----------------
@st.cache_resource
def load_predictor():
    """
    Loads ML models once and keeps them in memory.
    Prevents TensorFlow from reloading on every refresh.
    """
    return CattleWeightPredictor()

predictor = load_predictor()

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload a cattle image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- Prediction Logic ----------------
if uploaded_file is not None:

    # Open image safely
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Invalid image file.")
        st.text(str(e))
        st.stop()

    # Show uploaded image
    st.image(image, caption="Uploaded Image")

    # Predict button
    if st.button("Predict Weight"):

        with st.spinner("Analyzing cattle..."):

            try:
                weight, confidence = predictor.predict(image)

                # -------- Handle prediction results --------
                if weight is None:
                    st.error("Prediction failed. Check terminal for detailed error.")
                else:
                    st.success(f"Estimated Weight: {weight:.2f} kg")
                    st.info(f"Detection Confidence: {confidence:.2f}%")

            except Exception as e:
                st.error("An internal error occurred.")
                st.text(str(e))
                st.text(traceback.format_exc())

else:
    st.info("Please upload an image to start prediction.")
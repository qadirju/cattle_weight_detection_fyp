# Configuration file for Cattle Weight Detection

# Model normalization constants
# Replace these values with your actual training statistics
MEAN_WEIGHT = 300   # Replace with your actual mean weight from training data
STD_WEIGHT = 50     # Replace with your actual standard deviation from training data

# Image preprocessing constants
IMAGE_SIZE = (224, 224)

# Model architecture
MODEL_NAME = "efficientnet_b0"
MODEL_PATH = "best_model.pth"

# Streamlit configurations
APP_TITLE = "🐄 Cattle Weight Predictor"

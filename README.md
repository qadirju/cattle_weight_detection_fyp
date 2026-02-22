# cattle_weight_detection_fyp

## Overview
Vision-based cattle weight estimation using deep learning. Upload a cattle image and get an estimated weight using segmentation and regression models.

## Setup
1. Clone this repo and ensure your models are in `cattle-weight-detection-app/models/`:
	- `best_seg_model.keras`
	- `best_reg_model.keras`
2. Install dependencies:
	```bash
	pip install -r cattle-weight-detection-app/requirements.txt
	```
3. Run the Streamlit app:
	```bash
	streamlit run cattle-weight-detection-app/app.py
	```

## Usage
- Upload a cattle image (JPG, JPEG, PNG).
- The app will display the estimated weight and detection confidence.

## Troubleshooting
- **ModuleNotFoundError:** Run `pip install -r requirements.txt`.
- **Model load errors:** Ensure model files are present in the `models/` folder.
- **Prediction failed:** Check image format and model compatibility.

## Author
FYP Team

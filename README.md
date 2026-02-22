
# Cattle Weight Detection FYP

## Overview

Cattle Weight Detection FYP is a vision-based web application that estimates the weight of cattle from images using advanced deep learning techniques. The system leverages image segmentation and regression models to provide accurate weight predictions and detection confidence, making it a valuable tool for livestock management and research.

## Features

- 📷 Upload cattle images (JPG, JPEG, PNG)
- 🤖 Automated segmentation and weight estimation using Keras models
- 📊 Displays estimated weight and detection confidence
- 🖥️ User-friendly Streamlit web interface

## Requirements

- Python 3.10+
- Streamlit
- TensorFlow
- Pillow
- NumPy

All dependencies can be installed via the provided requirements file.

## Setup Instructions

1. **Clone the repository:**
	```bash
	git clone https://github.com/yourusername/cattle_weight_detection_fyp.git
	cd cattle_weight_detection_fyp
	```
2. **Add model files:**
	Place your trained models in `cattle-weight-detection-app/models/`:
	- `best_seg_model.keras` (Segmentation model)
	- `best_reg_model.keras` (Regression model)
3. **Create and activate a virtual environment (recommended):**
	```bash
	python -m venv .venv
	# On Windows:
	.venv\Scripts\activate
	# On macOS/Linux:
	source .venv/bin/activate
	```
4. **Install dependencies:**
	```bash
	pip install -r cattle-weight-detection-app/requirements.txt
	```
5. **Run the Streamlit app:**
	```bash
	streamlit run cattle-weight-detection-app/app.py
	```

## Usage

1. Open the web app in your browser (Streamlit will provide a local URL).
2. Upload a cattle image (JPG, JPEG, or PNG).
3. View the estimated weight and detection confidence on the results page.

## Troubleshooting

- **ModuleNotFoundError:** Ensure all dependencies are installed with `pip install -r cattle-weight-detection-app/requirements.txt`.
- **Model load errors:** Confirm that both model files are present in the `models/` directory.
- **Prediction failed:** Check that your image is a valid format and that the models are compatible with your data.
- **Streamlit not found:** Install it with `pip install streamlit` in your active environment.

## Project Structure

```
cattle_weight_detection_fyp/
├── cattle-weight-detection-app/
│   ├── app.py
│   ├── predictor.py
│   ├── requirements.txt
│   ├── models/
│   │   ├── best_seg_model.keras
│   │   └── best_reg_model.keras
│   └── sample_images/
└── README.md
```

## Credits

Developed by the FYP Team. For questions or contributions, please open an issue or pull request.

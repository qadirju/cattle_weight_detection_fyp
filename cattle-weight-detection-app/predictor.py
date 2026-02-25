import tensorflow as tf
import numpy as np
from PIL import Image
import os

class CattleWeightPredictor:
    """Segmentation + Regression predictor for cattle weight."""

    def __init__(self):

        # Dice coefficient (needed because model was trained with it)
        def dice_coef(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (
                tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
            )

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "models")

        seg_path = os.path.join(MODEL_DIR, "best_seg_model.keras")
        reg_path = os.path.join(MODEL_DIR, "best_reg_model.keras")

        # Load Segmentation Model
        try:
            print("Loading Segmentation Model from:", seg_path)
            self.seg_model = tf.keras.models.load_model(
                seg_path,
                custom_objects={"dice_coef": dice_coef}
            )
            print("Segmentation model loaded.")
        except Exception as e:
            print("❌ Segmentation model failed:", e)
            self.seg_model = None

        # Load Regression Model
        try:
            print("Loading Regression Model from:", reg_path)
            self.reg_model = tf.keras.models.load_model(reg_path)
            print("Regression model loaded.")
        except Exception as e:
            print("❌ Regression model failed:", e)
            self.reg_model = None

    # ---------------- PREPROCESS ----------------
    def preprocess(self, image):
        try:
            image = image.convert("RGB")
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            return image.astype("float32")
        except Exception as e:
            print("Preprocess error:", e)
            return None

    # ---------------- PREDICT ----------------
    def predict(self, image):

        if self.seg_model is None:
            print("Segmentation model not loaded")
            return None, None

        if self.reg_model is None:
            print("Regression model not loaded")
            return None, None

        img = self.preprocess(image)
        if img is None:
            return None, None

        try:
            img_batch = np.expand_dims(img, axis=0)

            # Segmentation
            mask = self.seg_model.predict(img_batch, verbose=0)[0]

            # Ensure mask is 2D (224, 224)
            if len(mask.shape) == 3:
                mask = np.squeeze(mask, axis=-1)

            # Expand mask to (224, 224, 1) for concatenation
            mask = np.expand_dims(mask, axis=-1)

            # Combine image + mask
            img_mask = np.concatenate([img, mask], axis=-1)
            img_mask_batch = np.expand_dims(img_mask, axis=0)

            # Regression
            weight = self.reg_model.predict(img_mask_batch, verbose=0)[0][0]

            # Confidence calculation
            cattle_pixels = np.sum(mask > 0.5)
            total_pixels = mask.size
            confidence = (cattle_pixels / total_pixels) * 100

            return float(weight), float(confidence)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None
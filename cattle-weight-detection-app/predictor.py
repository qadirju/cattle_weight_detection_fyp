import tensorflow as tf
import numpy as np
from PIL import Image

class CattleWeightPredictor:
    """Segmentation + Regression predictor for cattle weight."""

    def __init__(self):
        def dice_coef(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        try:
            self.seg_model = tf.keras.models.load_model(
                "models/best_seg_model.keras",
                custom_objects={"dice_coef": dice_coef}
            )
        except Exception as e:
            self.seg_model = None
            print(f"❌ Segmentation model load failed: {e}")

        try:
            self.reg_model = tf.keras.models.load_model("models/best_reg_model.keras")
        except Exception as e:
            self.reg_model = None
            print(f"❌ Regression model load failed: {e}")

        if self.seg_model and self.reg_model:
            print("✅ Models loaded successfully!")
        else:
            print("⚠️ One or both models failed to load.")

    def preprocess(self, image):
        try:
            image = image.resize((224,224))
            image = np.array(image)/255.0
            return image.astype("float32")
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None

    def predict(self, image):
        if not self.seg_model or not self.reg_model:
            return None, None
        img = self.preprocess(image)
        if img is None:
            return None, None
        try:
            img_batch = np.expand_dims(img,0)
            mask = self.seg_model.predict(img_batch, verbose=0)[0]
            img_mask = np.concatenate([img, mask], axis=-1)
            img_mask_batch = np.expand_dims(img_mask,0)
            weight = self.reg_model.predict(img_mask_batch, verbose=0)[0][0]
            # Confidence: proportion of mask pixels above threshold
            mask_threshold = 0.5
            cattle_pixels = np.sum(mask > mask_threshold)
            total_pixels = mask.size
            confidence = (cattle_pixels / total_pixels) * 100
            return float(weight), float(confidence)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            print("Please check your TensorFlow installation and model files.")
            return None, None
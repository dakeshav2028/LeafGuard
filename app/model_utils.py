import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image

class ModelHandler:
    """
    Handles loading the Keras model, preprocessing incoming images,
    and running inference to retrieve prediction confidence scores.
    """
    def __init__(self, model_path: str, classes_path: str):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.class_names = []
        
    def load_model(self):
        """
        Loads the pre-trained Keras model from disk and reads the class labels.
        Executed once during server startup (cold start mitigation).
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please run the training script first before starting the API."
            )
        if not os.path.exists(self.classes_path):
            raise FileNotFoundError(
                f"Classes label file not found at {self.classes_path}."
            )
            
        print(f"Loading Keras model from {self.model_path}...")
        # Load the compiled Keras model
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully.")
        
        # Load class labels
        with open(self.classes_path, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded class labels: {self.class_names}")
        
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Decodes raw image bytes, converts it to RGB format,
        resizes it to the expected 224x224 input shape, converts it to a numpy float array,
        and expands the dimensions to simulate a batch of size 1.
        """
        # Open image using Pillow
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure image is in RGB mode (handles PNGs with transparency or Grayscale images)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize using bilinear interpolation (matches training generator)
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Convert image to numpy float array
        img_array = np.array(image, dtype=np.float32)
        
        # Add a batch dimension: shape (224, 224, 3) becomes (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
        
    def predict(self, image_bytes: bytes) -> dict:
        """
        Processes raw image bytes, runs the image through the TensorFlow model,
        and returns structured predictions, including top class and alternative probabilities.
        """
        # Lazy load model if it hasn't been initialized
        if self.model is None:
            self.load_model()
            
        # Preprocess the raw bytes
        img_array = self.preprocess_image(image_bytes)
        
        # Perform inference. model.predict returns a 2D array of shape (batch_size, num_classes)
        # We index [0] to extract the predictions for our single input image.
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Identify the class with the highest probability
        predicted_idx = int(np.argmax(predictions))
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Rank predictions from highest to lowest confidence
        sorted_indices = np.argsort(predictions)[::-1]
        all_predictions = []
        for idx in sorted_indices:
            all_predictions.append({
                "class_name": self.class_names[idx],
                "confidence": float(predictions[idx])
            })
            
        # Return prediction details
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predictions": all_predictions
        }

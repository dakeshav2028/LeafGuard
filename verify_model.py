import os
import tensorflow as tf
import numpy as np

def verify():
    """
    Loads the trained model and performs a smoke test with a synthetic tensor
    to verify layer loading, inference shapes, and outputs.
    """
    model_path = "c:/MY_PROJECTS/Image_Classification_web_app/models/plant_disease_model.keras"
    classes_path = "c:/MY_PROJECTS/Image_Classification_web_app/evaluation/classes.txt"
    
    print("Checking model and class files...")
    assert os.path.exists(model_path), f"ERROR: Model not found at {model_path}"
    assert os.path.exists(classes_path), f"ERROR: Classes not found at {classes_path}"
    print("Files verified.")
    
    print("\nLoading saved Keras model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"Classes list: {classes}")
    
    print("\nCreating synthetic input tensor (shape: 1 x 224 x 224 x 3)...")
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    
    print("Running forward propagation...")
    predictions = model.predict(dummy_input, verbose=0)
    print("Forward pass completed.")
    
    # Assert correct output shape
    print(f"Predictions tensor shape: {predictions.shape}")
    assert predictions.shape == (1, 3), f"Expected shape (1, 3), but got {predictions.shape}"
    
    # Assert probability sums
    prob_sum = np.sum(predictions[0])
    print(f"Raw probabilities output: {predictions[0]}")
    print(f"Sum of output probabilities: {prob_sum:.6f}")
    assert np.isclose(prob_sum, 1.0, atol=1e-5), f"Expected probabilities to sum to ~1.0, got {prob_sum}"
    
    print("\nSUCCESS: The model loading and forward pass verified correctly!")

if __name__ == "__main__":
    verify()

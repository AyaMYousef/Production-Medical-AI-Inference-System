import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "autoencoder_model.keras")  
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold_full_dataset.pkl")  

# Load the model
model = load_model(MODEL_PATH)

with open(THRESHOLD_PATH, "rb") as f:
    THRESHOLD = pickle.load(f)

def predict_anomaly(image, threshold=0.05):
    """Run reconstruction and compute anomaly flag."""
    if threshold is None:
        threshold = THRESHOLD  

    image_batch = np.expand_dims(image, axis=0)
    start = time.time()
    # inference logic
    reconstructions = model.predict(image_batch)
    end = time.time()
    inference_time_ms = round((end - start) * 1000, 2)
    mse = np.mean((image - reconstructions[0]) ** 2)
    
    return {
        "reconstruction_mse": float(mse),
        "is_anomaly": bool(mse > threshold),
        "inference_time_ms":inference_time_ms
    }



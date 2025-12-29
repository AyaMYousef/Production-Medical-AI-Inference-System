from http.client import HTTPException
from app.model import predict_anomaly
from fastapi import FastAPI, UploadFile, File
from app.utils import preprocess_dicom, preprocess_image
import os
import pickle

app = FastAPI(title="DICOM Autoencoder Anomaly Detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold_full_dataset.pkl")  

with open(THRESHOLD_PATH, "rb") as f:
    THRESHOLD = pickle.load(f)


@app.get("/healtycheck")
def read_root():
    return {"Hello": "World"}

@app.get("/")
def root():
    return {"message": "Welcome to the DICOM Autoencoder API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...) ):
    """Upload a DICOM or image file and return anomaly prediction."""

    filename = file.filename.lower()

    # Detect file type
    if filename.endswith(".dcm"):
        img = preprocess_dicom(file.file)

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        img = preprocess_image(await file.read())

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload .dcm, .png, .jpg, .jpeg"
        )

    # Run prediction
    result = predict_anomaly(img, threshold=0.05)
    if "inference_time_ms" not in result:
        result["inference_time_ms"] = None
    return result
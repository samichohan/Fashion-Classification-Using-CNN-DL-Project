import io
import csv
import os
from datetime import datetime

import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ---------------- #
MODEL_PATH = "model/fashion_model.h5"
IMG_SIZE = (28, 28)
MAX_IMAGE_SIZE_MB = 5
LOG_PATH = "data/prediction_logs.csv"

CLASS_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# ---------------- APP INIT ---------------- #
app = FastAPI(
    title="Fashion MNIST Classifier API",
    description="Production API for classifying clothing images into 10 fashion categories using a CNN model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL (once, at startup) ---------------- #
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"⚠️ Failed to load model at startup: {e}")


# ---------------- REQUEST SCHEMA ---------------- #
class ImageURLRequest(BaseModel):
    url: str


# ---------------- HELPER FUNCTIONS ---------------- #
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert to grayscale, resize to 28x28, normalize, and reshape for the model."""
    img = img.convert("L")  # grayscale
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def run_prediction(img: Image.Image):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    arr = preprocess_image(img)
    probabilities = model.predict(arr)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = float(probabilities[predicted_index])

    # Top-3 predictions for richer UI display
    top3_indices = probabilities.argsort()[-3:][::-1]
    top3 = [
        {"label": CLASS_LABELS[int(i)], "confidence": round(float(probabilities[i]), 4)}
        for i in top3_indices
    ]

    return predicted_label, round(confidence, 4), top3


def log_prediction(source: str, predicted_label: str, confidence: float):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "predicted_label", "confidence"])
        writer.writerow([datetime.utcnow().isoformat(), source, predicted_label, confidence])


# ---------------- ROUTES ---------------- #
@app.get("/")
def root():
    return {"message": "Fashion MNIST Classifier API is running.", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None
    }


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"Image too large ({size_mb:.1f}MB). Max allowed: {MAX_IMAGE_SIZE_MB}MB.")

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    predicted_label, confidence, top3 = run_prediction(img)
    log_prediction(source="upload", predicted_label=predicted_label, confidence=confidence)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top3": top3
    }


@app.post("/predict/url")
def predict_url(request: ImageURLRequest):
    try:
        response = requests.get(request.url, timeout=10, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Could not fetch image from the provided URL.")

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise HTTPException(status_code=400, detail="The URL does not point to a valid image.")

    content = response.content
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"Image too large ({size_mb:.1f}MB). Max allowed: {MAX_IMAGE_SIZE_MB}MB.")

    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open the image from the URL.")

    predicted_label, confidence, top3 = run_prediction(img)
    log_prediction(source="url", predicted_label=predicted_label, confidence=confidence)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top3": top3
    }

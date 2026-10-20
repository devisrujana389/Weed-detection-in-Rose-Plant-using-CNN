"""
Weed Detection API (FastAPI + TensorFlow)

Run from the backend_api folder:
    uvicorn main:app --host 0.0.0.0 --port 5000 --reload

Endpoints:
    GET  /health   -> is the model loaded, what input size / scaling is used
    POST /predict  -> multipart/form-data, field name "file" (an image)
"""
import io
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------- settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(BASE_DIR, "..", "models", "weed_detector.h5")
)
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))       # weed if prob >= this
INVERT_LABELS = os.getenv("INVERT_LABELS", "0") == "1"  # set 1 if labels reversed
SCALE_MODE = os.getenv("SCALE_MODE", "auto")            # auto | raw | div255
MAX_MB = 10

# ---------------------------------------------------------------- load model
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# input size is read from the model itself, so no manual IMG_SIZE edits needed
_, H, W, _ = model.input_shape
IMG_SIZE = (W, H)  # PIL uses (width, height)


def _has_rescaling(layer) -> bool:
    """True if the model already has a Rescaling layer (expects 0-255 pixels)."""
    if isinstance(layer, tf.keras.layers.Rescaling):
        return True
    return any(_has_rescaling(sub) for sub in getattr(layer, "layers", []))


if SCALE_MODE == "raw":
    SCALE_IN_API = False
elif SCALE_MODE == "div255":
    SCALE_IN_API = True
else:  # auto
    SCALE_IN_API = not _has_rescaling(model)

print(f"[API] model={MODEL_PATH}")
print(f"[API] input size={IMG_SIZE}, divide by 255 in API={SCALE_IN_API}, "
      f"invert labels={INVERT_LABELS}, threshold={THRESHOLD}")

# ---------------------------------------------------------------- app
app = FastAPI(title="Weed Detection in Rose Plants - CNN API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def preprocess(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    if SCALE_IN_API:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "input_size": list(IMG_SIZE),
        "divide_by_255_in_api": SCALE_IN_API,
        "invert_labels": INVERT_LABELS,
        "threshold": THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image larger than {MAX_MB} MB.")

    batch = preprocess(data)
    weed_prob = float(model.predict(batch, verbose=0)[0][0])
    if INVERT_LABELS:
        weed_prob = 1.0 - weed_prob

    is_weed = weed_prob >= THRESHOLD
    confidence = weed_prob if is_weed else 1.0 - weed_prob
    return {
        "prediction": "Weed Detected" if is_weed else "No Weed Detected",
        "confidence": round(confidence, 4),
        "weed_probability": round(weed_prob, 4),
    }
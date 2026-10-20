"""
Run prediction on a single image.

Usage:
    python src/predict.py --image path/to/photo.jpg --model_path models/weed_detector.h5
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import load_image, resize_image


def predict_image(image_path: str, model_path: str = "models/weed_detector.h5", threshold: float = 0.5):
    model = load_model(model_path)

    img = load_image(image_path)
    img = resize_image(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prob = float(model.predict(img, verbose=0)[0][0])
    label = "Weed Detected" if prob > threshold else "No Weed Detected"
    confidence = prob if prob > threshold else 1 - prob

    return label, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/weed_detector.h5")
    args = parser.parse_args()

    label, confidence = predict_image(args.image, args.model_path)
    print(f"\nPrediction: {label}  (confidence: {confidence*100:.2f}%)")


if __name__ == "__main__":
    main()

"""
Streamlit demo app for the Weed Detection in Rose Plants project.

Run with:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tensorflow.keras.models import load_model  # noqa: E402

st.set_page_config(page_title="Rose Plant Weed Detector", page_icon="🌹", layout="centered")

MODEL_PATH = "models/weed_detector.h5"


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


def predict(img: Image.Image, model):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "🌿 Weed Detected" if prob > 0.5 else "🌹 No Weed Detected"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence


st.title("🌹 Weed Detection in Rose Plants")
st.caption("CNN-based weed classification — upload a photo of a rose plant to check for weeds.")

model = get_model()

if model is None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Train one first:\n\n"
        "```bash\npython src/train.py --data_dir data/raw --model_type mobilenet --epochs 20\n```"
    )
else:
    uploaded = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            label, confidence = predict(image, model)

        st.subheader(label)
        st.progress(min(int(confidence * 100), 100))
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
    else:
        st.info("Upload a .jpg or .png image to get a prediction.")

st.divider()
st.caption("Weed Detection in Rose Plants using CNN — Sir MVIT, Dept. of ISE, 2025-2026")

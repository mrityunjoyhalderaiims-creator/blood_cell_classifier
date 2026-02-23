import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Blood Cell Classifier", layout="centered")

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"
IMG_SIZE = (224, 224)  # Teachable Machine image models typically use 224x224

# ----------------------------
# Helpers
# ----------------------------
def clean_label(label: str) -> str:
    """
    Converts '0 Basophil' -> 'Basophil'
    Keeps original text if no numeric prefix exists.
    """
    label = label.strip()
    parts = label.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1].strip()
    return label


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocess image for Teachable Machine Keras model.
    Expected: float32, shape (1, 224, 224, 3)
    Normalization: (img / 127.5) - 1
    """
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, img: Image.Image):
    input_data = preprocess_image(img)
    prediction = model.predict(input_data, verbose=0)  # shape: (1, n_classes)

    # Some models already output probabilities, some may output logits
    probs = prediction[0].astype(float)

    # If values do not sum close to 1, apply softmax
    s = probs.sum()
    if not np.isfinite(s) or s <= 0 or not (0.98 <= s <= 1.02):
        exp_vals = np.exp(probs - np.max(probs))
        probs = exp_vals / exp_vals.sum()

    pred_index = int(np.argmax(probs))
    pred_conf = float(probs[pred_index])

    return probs, pred_index, pred_conf


# ----------------------------
# UI
# ----------------------------
st.title("Blood Cell Classifier")

# Load model + labels safely
try:
    model = load_model()
    labels = load_labels()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        try:
            probs, pred_index, pred_conf = predict_image(model, image)

            # Safety check in case labels and outputs mismatch
            n_classes_model = len(probs)
            n_classes_labels = len(labels)

            if n_classes_model != n_classes_labels:
                st.warning(
                    f"Warning: Model outputs {n_classes_model} classes but labels.txt has {n_classes_labels} labels."
                )

            # Main prediction display
            pred_label_raw = labels[pred_index] if pred_index < len(labels) else f"Class {pred_index}"
            pred_label = clean_label(pred_label_raw)

            st.subheader("Prediction")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted class", pred_label)

            with col2:
                st.metric("Confidence", f"{pred_conf * 100:.2f}%")

            st.progress(float(pred_conf))

            # Top-2 predictions
            st.subheader("Top predictions")
            top_k = min(2, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k]

            for rank, idx in enumerate(top_indices, start=1):
                label_raw = labels[idx] if idx < len(labels) else f"Class {idx}"
                label_clean = clean_label(label_raw)
                prob = float(probs[idx])

                st.write(f"**{rank}. {label_clean}** — {prob * 100:.2f}%")
                st.progress(prob)

            # All class probabilities
            st.subheader("All class probabilities")
            for i, prob in enumerate(probs):
                label_raw = labels[i] if i < len(labels) else f"Class {i}"
                label_clean = clean_label(label_raw)

                st.write(f"{label_clean}: {float(prob) * 100:.2f}%")
                st.progress(float(prob))

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.info("Upload an image to begin.")

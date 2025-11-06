# app.py
import os, json
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effv2_preprocess

working_dir = Path(__file__).parent

mobile_net_path   = working_dir / 'MobileNet.h5'
efficientnet_path = working_dir / 'EfficientNetV2B0.h5'
class_indices_path = working_dir / 'class_indices.json'

try:
    if not mobile_net_path.exists():
        st.error(f"File model tidak ditemukan: {mobile_net_path}")
        st.stop()
    if not efficientnet_path.exists():
        st.error(f"File model tidak ditemukan: {efficientnet_path}")
        st.stop()

    mobile_net_model   = tf.keras.models.load_model(mobile_net_path, compile=False)
    efficientnet_model = tf.keras.models.load_model(efficientnet_path, compile=False)
except Exception as e:
    st.error("Failed to load the models. Periksa file .h5 dan versi TensorFlow/Keras.")
    st.exception(e)
    st.stop()

try:
    with class_indices_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if all(isinstance(v, int) for v in mapping.values()):  # name->idx
        idx2name = {str(v): k for k, v in mapping.items()}
    else:                                                  # idx->name
        idx2name = {str(k): v for k, v in mapping.items()}
except Exception as e:
    st.error(f"Gagal membaca class_indices.json di: {class_indices_path}")
    st.exception(e)
    st.stop()


def load_and_preprocess_image(file_like, target_size=(224, 224), mode="rescale"):
    try:
        file_like.seek(0)
    except Exception:
        pass

    img = Image.open(file_like).convert("RGB")
    img = img.resize(target_size)
    x = np.asarray(img, dtype=np.float32)

    if mode == "effv2":
        x = effv2_preprocess(x)
    else:
        x = x / 255.0

    x = np.expand_dims(x, axis=0)
    return x

def predict_image_class(model, file_like, idx2name, preprocess_mode="rescale"):
    x = load_and_preprocess_image(file_like, mode=preprocess_mode)
    preds = model.predict(x, verbose=0)[0]
    top = int(np.argmax(preds))
    label = idx2name.get(str(top), f"class_{top}")
    conf  = float(preds[top])
    return label, conf

st.title("Klasifikasi Spesies Burung Berbasis Citra")
st.text(
    "Model ini mengklasifikasikan spesies burung dari citra/foto. Tujuannya mendukung konservasi, pemantauan keanekaragaman hayati, dan edukasi "
    "citizen science dengan identifikasi cepat dan akurat langsung dari gambar lapangan. Tantangan utama meliputi kemiripan morfologi antargatun, variasi "
    "pose & latar, serta pencahayaan yang beragam; untuk itu model memanfaatkan transfer learning EfficientNetV2-B0 dan MobileNetV2 disertai dengan augmentasi "
    "data dan fine-tuning agar robust."
)

st.subheader("Berikut adalah spesies burung yang bisa dideteksi: ")

st.subheader("Pilih Algoritma")

model_choice = st.selectbox(
    "Pilih Model yang Akan Digunakan:",
    ("MobileNetV2", "EfficientNetV2B0")
)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button("Classify"):
            if model_choice == "MobileNetV2":
                model = mobile_net_model
                preprocess_mode = "rescale"
            else:
                model = efficientnet_model
                preprocess_mode = "effv2"

            label, conf = predict_image_class(model, uploaded_image, idx2name, preprocess_mode)
            st.success(f"Prediction: **{label}** ({conf:.2%})")

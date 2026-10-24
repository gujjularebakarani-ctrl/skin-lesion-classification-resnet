import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "skin_cancer_model.keras")

st.set_page_config(page_title="Skin Lesion Classification")
st.title("Skin Lesion Classification")
st.write("Upload a skin image for model prediction.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}\n\n"
             "Place skin_cancer_model.keras in the same folder as app.py.")
    st.stop()

uploaded_file = st.file_uploader("Choose a skin image",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        try:
            model = load_model()
        except Exception as e:
            st.error("Could not load the model. Keras/TensorFlow version may "
                     "be older than 3.13 - run: pip install -U tensorflow keras")
            st.exception(e)
            st.stop()

        # model already contains ResNet50 preprocess_input -> feed raw 0-255 pixels
        arr = np.array(image.resize((224, 224)), dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        score = float(model.predict(arr, verbose=0).reshape(-1)[0])

        if score >= 0.5:
            st.error("Predicted class: MALIGNANT")
            st.write(f"Model confidence: {score:.2%}")
        else:
            st.success("Predicted class: BENIGN")
            st.write(f"Model confidence: {(1 - score):.2%}")

st.caption("Academic demo only - not a medical diagnosis.")

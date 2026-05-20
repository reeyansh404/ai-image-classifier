import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image


@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


def preprocess_image(image):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def classify_image(model, image):
    try:
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        decoded = decode_predictions(predictions, top=3)[0]
        return decoded
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None


def main():
    st.set_page_config(
        page_title="AI Image Classifier",
        page_icon="🖼️",
        layout="centered"
    )

    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing image..."):
                image = Image.open(uploaded_file).convert("RGB")
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")


if __name__ == "__main__":
    main()
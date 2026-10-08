from typing import NamedTuple
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Image Detection", page_icon="📷", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

net = load_model()

CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Other Corruption", "Pothole"]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Image")
st.write("Upload an image and start detecting.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if image_file is not None:
    image = Image.open(image_file)
    col1, col2 = st.columns(2)

    _image = np.array(image)
    results = net.predict(_image, conf=score_threshold)
    
    annotated_frame = results[0].plot()
    _image_pred = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    with col1:
        st.write("#### Original Image")
        st.image(_image)
    
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        
        st.download_button(
            label="Download Prediction Image",
            data=buffer.getvalue(),
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
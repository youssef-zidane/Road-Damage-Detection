import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="Realtime Detection", page_icon="📷", layout="centered")

# استخدام خادم STUN مجاني من جوجل لضمان عمل الكاميرا
STUN_SERVER = [{"urls": ["stun:stun.l.google.com:19302"]}]

@st.cache_resource
def load_model():
    return YOLO("best.pt")

net = load_model()

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Realtime")
st.write("Detect road damage in realtime using your Webcam.")

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h_ori, w_ori = image.shape[:2]
    
    results = net.predict(image, conf=score_threshold, verbose=False)
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
           Detection(
               class_id=int(_box.cls),
               label=CLASSES[int(_box.cls)],
               score=float(_box.conf),
               box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
        result_queue.put(detections)

    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase if there is false prediction.")

webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
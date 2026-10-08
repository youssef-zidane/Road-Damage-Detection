import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Video Detection", page_icon="🎥", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

net = load_model()

# إنشاء مجلد temp لو مش موجود لحفظ الفيديو المعالج
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    write_bytesio_to_file(temp_file_input, video_file)
    videoCapture = cv2.VideoCapture(temp_file_input)

    if not videoCapture.isOpened():
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

        st.write(f"Resolution: {_width}x{_height} | FPS: {_fps} | Total Frames: {_frame_count}")

        inferenceBar = st.progress(0, text="Performing inference on video, please wait.")
        imageLocation = st.empty()

        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        _frame_counter = 0
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()
            if ret:
                results = net.predict(frame, conf=score_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                cv2writer.write(annotated_frame)
                
                # عرض الفريمات الحية أثناء المعالجة
                imageLocation.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

                _frame_counter += 1
                inferenceBar.progress(_frame_counter / _frame_count, text=f"Processing frame {_frame_counter}/{_frame_count}")
            else:
                inferenceBar.empty()
                break

        videoCapture.release()
        cv2writer.release()

    st.success("Video Processed Successfully!")

    with open(temp_file_infer, "rb") as f:
        st.download_button(
            label="Download Prediction Video",
            data=f,
            file_name="RDD_Prediction.mp4",
            mime="video/mp4",
            use_container_width=True
        )

st.title("Road Damage Detection - Video")
st.write("Upload the video and start detecting.")

video_file = st.file_uploader("Upload Video", type=".mp4")
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if video_file is not None:
    if st.button('Process Video', use_container_width=True, type="primary"):
        st.warning(f"Processing Video {video_file.name}")
        processVideo(video_file, score_threshold)
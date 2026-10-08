import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
)

st.title("Road Damage Detection Application")

st.markdown(
    """
    Introducing our Road Damage Detection Apps, powered by the YOLO11 deep learning model.
    
    This application is designed to enhance road safety and infrastructure maintenance by swiftly identifying and categorizing various forms of road damage.

    There are five types of damage that this model can detect:
    - Longitudinal Crack
    - Transverse Crack
    - Alligator Crack
    - Other Corruption
    - Pothole

    You can select the apps from the sidebar to try and experiment with any kind of input **(realtime-webcam, video, and images)** depending on your use case.
    """
)
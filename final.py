import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import streamlit as st
import mediapipe as mp
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Paths
database_path = 'Database'
attendance_file = 'attendance.csv'

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize the attendance log (if not exists)
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

def recognition(verification_image):  
    identity_verify = DeepFace.find(img_path=verification_image, db_path=database_path, model_name='Facenet512')
    try:
        found_image_path = identity_verify[0]['identity'][0].split('/')
        verified_person = found_image_path[-2]
        return verified_person, identity_verify[0]['identity'][0]
    except:
        return None, None

def detect_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    return "green" if results.detections else "red"

def log_attendance(user_name):
    current_time = datetime.now()
    date = current_time.date()
    time_str = current_time.strftime("%H:%M:%S")
    
    df = pd.read_csv(attendance_file)
    new_df = pd.DataFrame({"Name": [user_name], "Date": [date], "Time": [time_str]})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(attendance_file, index=False)

# WebRTC Video Processor
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        border_color = detect_face(img)
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0) if border_color == "green" else (0, 0, 255))
        return img

# Streamlit UI
st.title("Face Recognition System")
st.subheader("Align your face and validate")

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoProcessor)

if webrtc_ctx.video_transformer:
    st.write("Webcam Stream Active")

# Attendance Page
def attendance_page():
    st.title("Attendance Records")
    df = pd.read_csv(attendance_file)
    if df.empty:
        st.write("No attendance records available.")
    else:
        st.write(df)

# Sidebar Navigation
page = st.sidebar.radio("Choose a page", ["Face Recognition", "Attendance"])
if page == "Attendance":
    attendance_page()

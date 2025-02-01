import streamlit as st
import cv2
from deepface import DeepFace
import os
import pandas as pd
from datetime import datetime
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize the attendance log (if not exists)
attendance_file = 'attendance.csv'

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# Function to log attendance
def log_attendance(user_name):
    current_time = datetime.now()
    date = current_time.date()
    time_str = current_time.strftime("%H:%M:%S")
    
    # Append the attendance to the CSV
    df = pd.read_csv(attendance_file)
    new_df = pd.DataFrame({"Name": [user_name], "Date": [date], "Time": [time_str]})
    
    # Combine the DataFrames by appending rows
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(attendance_file, index=False)

# Define a class to handle video frame transformation
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detected = False
        self.face_detected_time = None
        self.countdown = 5  # seconds

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format

        # Detect face (simplified example with Haar cascade, you can use DeepFace here)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if faces != ():
            self.face_detected = True
            if self.face_detected_time is None:
                self.face_detected_time = time.time()

            # Draw rectangle around the face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Start timer when face is detected
            elapsed_time = time.time() - self.face_detected_time
            if elapsed_time >= self.countdown:
                # After 5 seconds, you can perform recognition
                self.face_detected = False  # Reset the face detection flag
                user_name = "Verified Person"  # Replace with actual recognition logic

                # Log attendance (you can replace with actual DeepFace recognition)
                log_attendance(user_name)

                st.write(f"Attendance verified for: {user_name}")
                
        # Return the processed frame
        return img

# Streamlit UI setup
st.title("Face Recognition System with Streamlit-WebRTC")

# Start the Streamlit WebRTC stream and video transformer
webrtc_streamer(key="face-recognition", video_transformer_factory=VideoTransformer)

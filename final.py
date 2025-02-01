import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import streamlit as st
import mediapipe as mp
import cv2

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

# Function to recognize face
def recognition(verification_image):  
    identity_verify = DeepFace.find(img_path=verification_image, db_path=database_path, model_name='Facenet512')
    try:
        found_image_path = identity_verify[0]['identity'][0].split('/')
        verified_person = found_image_path[-2]
        return verified_person, identity_verify[0]['identity'][0]
    except:
        return None, None

# Function to log attendance
def log_attendance(user_name):
    current_time = datetime.now()
    date = current_time.date()
    time_str = current_time.strftime("%H:%M:%S")
    
    df = pd.read_csv(attendance_file)
    new_df = pd.DataFrame({"Name": [user_name], "Date": [date], "Time": [time_str]})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(attendance_file, index=False)

# Streamlit UI
st.title("Face Recognition System")
st.subheader("Upload an Image for Recognition")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)
    
    # Perform recognition
    verified_name, image_path = recognition(file_path)
    
    if verified_name:
        log_attendance(verified_name)
        st.success(f"Attendance Verified for {verified_name}!")
        st.image(image_path, caption="Matched Image", use_column_width=True)
    else:
        st.error("User not found in the database!")

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

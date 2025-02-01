import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import streamlit as st
import mediapipe as mp
import cv2
import time

# Paths
test = '/Users/sainikheel/Personal/Face_recognition/Database/User-1/User-1.jpg'
database_path = 'Database'
attendance_file = 'attendance.csv'  # CSV file to store attendance logs

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
    if results.detections:
        return "green"
    return "red"

def cropped_frame(frame):
    crop_width, crop_height = 1000, 1000
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate the center of the frame
    center_x, center_y = frame_width // 2, frame_height // 2

    # Calculate the cropping box around the center
    crop_x1 = max(center_x - crop_width // 2, 0)
    crop_y1 = max(center_y - crop_height // 2, 0)
    crop_x2 = min(center_x + crop_width // 2, frame_width)
    crop_y2 = min(center_y + crop_height // 2, frame_height)

    # Crop the frame
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    return cropped_frame

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

# Streamlit UI
st.title("Face Recognition System")
st.subheader("Align your face and validate")

start_webcam = st.button('Start Webcam')
frame_holder = st.empty()
status_text = st.empty()
validate_user = ""
frame = None

if start_webcam:
    cap = cv2.VideoCapture(0)
        
    # Timer placeholder
    timer_placeholder = st.empty()
    countdown = 5  # 10-second countdown after face detection
    
    face_detected = False  # Flag to track when the face is detected
    face_detected_time = None  # Timestamp when face is detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam!")
        
        frame = cropped_frame(frame)

        # Detect face and set border color
        border_color = detect_face(frame)
        
        # Draw border
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0) if border_color == "green" else (0, 0, 255))
        frame_holder.image(frame, channels="BGR")
        
        if border_color == 'green' and not face_detected:
            face_detected = True
            face_detected_time = time.time()  # Record the time when the face is detected
            status_text.write("Status: Face detected! A 5-second timer will start now.")
        
        if face_detected:
            # Calculate the elapsed time since face detection
            elapsed_time = time.time() - face_detected_time
            
            if elapsed_time < countdown:
                remaining_time = countdown - int(elapsed_time)
                timer_placeholder.write(f"Timer: {remaining_time} seconds remaining...")
            else:
                verified_name, image_path = recognition(frame)
                if verified_name:
                    log_attendance(verified_name)
                    with st.expander(f"Attendance Verified for {verified_name}"):
                        st.write(f"Hello, {verified_name}!")
                        frame_holder_database = st.empty()
                        frame_holder_database.image(image_path, channels='BGR')
                else:
                    st.write('User is not found in the database!')
                break

# Attendance Page
def attendance_page():
    st.title("Attendance Records")
    
    # Read the attendance log CSV file
    df = pd.read_csv(attendance_file)
    
    # Show the attendance log
    if df.empty:
        st.write("No attendance records available.")
    else:
        for index, row in df.iterrows():
            st.write(f"{row['Name']} | Date: {row['Date']} | Time: {row['Time']}")

# Sidebar Navigation
page = st.sidebar.radio("Choose a page", ["Face Recognition", "Attendance"])

if page == "Face Recognition":
    # Include the Face Recognition UI here
    pass
else:
    attendance_page()

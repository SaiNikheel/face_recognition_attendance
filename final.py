import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import streamlit as st
import cv2
import numpy as np
import pytz

# Paths
database_path = 'Database'
attendance_file = 'attendance.csv'

# Initialize the attendance log (if not exists)
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# Function to recognize face
def recognition(verification_image):  
    try:
        identity_verify = DeepFace.find(img_path=verification_image, anti_spoofing=True, db_path=database_path, model_name='Facenet512')
        found_image_path = identity_verify[0]['identity'][0].split('/')
        verified_person = found_image_path[-2]
        return verified_person, identity_verify[0]['identity'][0]
    except:
        return None, None

# Function to log attendance
def log_attendance(user_name):
    current_time = datetime.now()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    date = current_time.date()
    time_str = current_time.strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)
    new_df = pd.DataFrame({"Name": [user_name], "Date": [date], "Time": [time_str]})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(attendance_file, index=False)

# Streamlit UI
st.title("Face Recognition System")
st.subheader("Capture an Image for Recognition")

# Capture image from camera
camera_image = st.camera_input("Take a picture")

if camera_image:
    file_path = "temp_image.jpg"

    # Convert to OpenCV format
    image_bytes = camera_image.getvalue()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save the image
    cv2.imwrite(file_path, img)

    # Display captured image
    st.image(file_path, caption="Captured Image", use_container_width=True)

    # Perform recognition
    verified_name, image_path = recognition(file_path)

    if verified_name:
        log_attendance(verified_name)
        st.success(f"Attendance Verified for {verified_name}!")
        st.image(image_path, caption="Matched Image", use_container_width=True)
    else:
        st.error("User not found in the database!, Or no Live person on the picture.")

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

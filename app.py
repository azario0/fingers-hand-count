import streamlit as st  # Streamlit for creating the web app
import cv2  # OpenCV for accessing the webcam and processing images
import mediapipe as mp  # MediaPipe for hand tracking
import numpy as np  # NumPy for numerical operations (not explicitly used in this snippet)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands  # Access MediaPipe's hand tracking solution
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)  # Initialize hand tracking
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing hand landmarks

# Streamlit app
st.title("Finger Count using MediaPipe")  # Set the title of the Streamlit app

# Function to count fingers
def count_fingers(image, results):
    height, width, _ = image.shape
    fingers = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * width)
                lmy = int(lm.y * height)
                landmarks.append((lmx, lmy))

            # Thumb
            if landmarks[mp_hands.HandLandmark.THUMB_TIP][0] < landmarks[mp_hands.HandLandmark.THUMB_IP][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other four fingers
            for id in range(1, 5):
                if landmarks[mp_hands.HandLandmark(id*4+4)][1] < landmarks[mp_hands.HandLandmark(id*4+2)][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

    return fingers.count(1)

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Access the webcam

# Run the app
if st.checkbox('Run'):  # Checkbox to start/stop the webcam feed
    st_frame = st.empty()  # Placeholder for the video frame
    st_fingers = st.empty()  # Placeholder for the finger count

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            st.write("Unable to access camera")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
        results = hands.process(frame)  # Process the frame for hand tracking

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw hand landmarks on the frame

        fingers_count = count_fingers(frame, results)  # Count the number of extended fingers
        st_frame.image(frame, channels='RGB')  # Display the frame in the Streamlit app
        st_fingers.text(f'Fingers: {fingers_count}')  # Display the finger count in the Streamlit app

cap.release()  # Release the webcam
hands.close()  # Close the MediaPipe hand tracking

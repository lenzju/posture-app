import streamlit as st
import cv2
import mediapipe as mp

st.title("Posture Detection (MediaPipe)")

run = st.checkbox("Start Kamera")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Keine Kamera gefunden")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        FRAME_WINDOW.image(frame)

cap.release()

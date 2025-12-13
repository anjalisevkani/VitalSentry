import streamlit as st
import cv2
import numpy as np
from src.camera import get_camera
from src.face_mesh import face_mesh
from src.roi_extractor import extract_roi
from src.signal_processing import bandpass
from src.heart_rate import calculate_bpm, calculate_respiration

st.title("💓 VitalSentry – Contactless Heart & Respiration Monitor")

# Config
FOREHEAD = [10, 338, 297]
fs = 30  # camera FPS

# Initialize session state
if "green_signal" not in st.session_state:
    st.session_state.green_signal = []
if "bpm_list" not in st.session_state:
    st.session_state.bpm_list = []

# Placeholder for images
camera_placeholder = st.empty()
roi_placeholder = st.empty()
bpm_placeholder = st.empty()
rr_placeholder = st.empty()

# Start camera
cap = get_camera()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        forehead = extract_roi(frame, landmarks, FOREHEAD)

        if forehead.size > 0:
            st.session_state.green_signal.append(np.mean(forehead[:, :, 1]))

            # Filter and BPM calculation
            if len(st.session_state.green_signal) >= 150:
                filtered_signal = bandpass(st.session_state.green_signal, fs)
                bpm = calculate_bpm(filtered_signal, fs)
                st.session_state.bpm_list.append(bpm)
                if len(st.session_state.bpm_list) > 5:
                    st.session_state.bpm_list.pop(0)
                bpm_smooth = int(np.mean(st.session_state.bpm_list))
                bpm_placeholder.markdown(f"**Heart Rate (BPM): {bpm_smooth}**")

            # Respiration Rate
            if len(st.session_state.green_signal) >= 300:
                rr = calculate_respiration(st.session_state.green_signal, fs)
                rr_placeholder.markdown(f"**Respiration Rate (RR): {int(rr)}**")

            # Show ROI
            roi_placeholder.image(forehead, channels="BGR", width=200)

    # Show camera frame
    camera_placeholder.image(frame, channels="BGR", width=400)

# Release camera when done
cap.release()

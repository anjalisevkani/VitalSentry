import cv2
import numpy as np
from src.camera import get_camera
from src.face_mesh import face_mesh
from src.roi_extractor import extract_roi
from src.signal_processing import bandpass
from src.heart_rate import calculate_bpm, calculate_respiration

# -------------------------------
# Configuration
# -------------------------------
FOREHEAD = [10, 338, 297]  # Landmarks for forehead ROI
fs = 30  # Camera frames per second
cap = get_camera()

# -------------------------------
# Signal storage
# -------------------------------
green_signal = []  # raw green channel signal
bpm_list = []      # rolling average for heart rate

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # STEP 7: Extract Forehead ROI
        forehead = extract_roi(frame, landmarks, FOREHEAD)

        if forehead.size > 0:
            cv2.imshow("Forehead ROI", forehead)

            # STEP 8: Extract green channel mean
            green_mean = np.mean(forehead[:, :, 1])
            green_signal.append(green_mean)

            # STEP 9: Filter signal if enough frames
            if len(green_signal) >= 150:
                filtered_signal = bandpass(green_signal, fs)

                # STEP 10: Heart Rate Calculation
                bpm = calculate_bpm(filtered_signal, fs)
                bpm_list.append(bpm)
                if len(bpm_list) > 5:
                    bpm_list.pop(0)
                bpm_smooth = int(np.mean(bpm_list))

                cv2.putText(frame, f"BPM: {bpm_smooth}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            # STEP 11: Respiration Rate Calculation
            if len(green_signal) >= 300:  # need ~10 sec of data
                rr = calculate_respiration(green_signal, fs)
                rr_smooth = int(rr)
                cv2.putText(frame, f"RR: {rr_smooth} bpm",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

    # Display main camera feed
    cv2.imshow("VitalSentry", frame)

    # Exit on 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# Release resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()

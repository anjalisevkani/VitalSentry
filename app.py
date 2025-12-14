import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fftpack import fft
import time

# ===============================
# CONFIGURATION
# ===============================
fs = 30  # Camera FPS
WINDOW_SECONDS = 8
WINDOW_SIZE = fs * WINDOW_SECONDS

RR_WINDOW_SECONDS = 20
RR_WINDOW_SIZE = fs * RR_WINDOW_SECONDS

alpha_bpm = 0.2
alpha_rr = 0.1

# ===============================
# SIGNAL PROCESSING FUNCTIONS
# ===============================
def bandpass(signal, fs, low=0.7, high=4.0):
    nyq = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def calculate_bpm(signal, fs):
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(fft_vals), 1/fs)

    mask = (freqs > 0.7) & (freqs < 4.0)
    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    return peak_freq * 60

def calculate_respiration(signal, fs):
    filtered = bandpass(signal, fs, low=0.1, high=0.5)
    peaks, _ = find_peaks(filtered, distance=fs*2)
    duration = len(signal) / fs
    return (len(peaks) / duration) * 60

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False)

FOREHEAD = [10, 338, 297, 332]
CHEEKS = [50, 280]

# ===============================
# VARIABLES
# ===============================
green_signal = []
smoothed_bpm = None
smoothed_rr = None
frame_count = 0

# ===============================
# VIDEO CAPTURE
# ===============================
cap = cv2.VideoCapture(0)

print("[INFO] VitalSentry Started... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        roi_pixels = []

        for idx in FOREHEAD + CHEEKS:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            roi_pixels.append(frame[y, x, 1])  # Green channel

        mean_green = np.mean(roi_pixels)
        green_signal.append(mean_green)

        # ===============================
        # HEART RATE CALCULATION
        # ===============================
        if len(green_signal) >= WINDOW_SIZE and frame_count % fs == 0:
            window_signal = green_signal[-WINDOW_SIZE:]
            filtered = bandpass(window_signal, fs)
            bpm = calculate_bpm(filtered, fs)

            if smoothed_bpm is None:
                smoothed_bpm = bpm
            else:
                smoothed_bpm = alpha_bpm * bpm + (1 - alpha_bpm) * smoothed_bpm

        # ===============================
        # RESPIRATION RATE CALCULATION
        # ===============================
        if len(green_signal) >= RR_WINDOW_SIZE and frame_count % (3 * fs) == 0:
            rr_signal = green_signal[-RR_WINDOW_SIZE:]
            rr = calculate_respiration(rr_signal, fs)

            if smoothed_rr is None:
                smoothed_rr = rr
            else:
                smoothed_rr = alpha_rr * rr + (1 - alpha_rr) * smoothed_rr

        # ===============================
        # DISPLAY
        # ===============================
        if smoothed_bpm:
            cv2.putText(frame, f"Heart Rate: {int(smoothed_bpm)} BPM",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        if smoothed_rr:
            cv2.putText(frame, f"Respiration Rate: {int(smoothed_rr)} BPM",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

    cv2.imshow("VitalSentry – Contactless Vitals Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()

import streamlit as st
import cv2
import numpy as np
from src.camera import get_camera
from src.face_mesh import face_mesh
from src.roi_extractor import extract_roi
from src.signal_processing import bandpass
from src.heart_rate import calculate_bpm, calculate_respiration

# ===============================
# 1. PAGE CONFIGURATION & CSS
# ===============================
st.set_page_config(
    page_title="VitalSentry Dashboard",
    page_icon="💓",
    layout="wide"
)

st.markdown("""
    <style>
        /* Remove top padding */
        .block-container {padding-top: 1rem;}
        
        /* Style the metric container */
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #464b5c;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }

        /* Label (Bright White) */
        div[data-testid="stMetricLabel"] > div {
            color: #FFFFFF !important;
            font-size: 16px !important;
            font-weight: bold;
        }
        
        /* Value (Neon Green) */
        div[data-testid="stMetricValue"] > div {
            color: #00FF41 !important;
            font-size: 32px !important;
            font-weight: bold;
        }

        /* Center images */
        div[data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 2. UI LAYOUT
# ===============================
# Sidebar
with st.sidebar:
    st.title("💓 VitalSentry")
    st.markdown("---")
    run_monitoring = st.toggle("Start Monitoring", value=False)
    
    st.markdown("### ROI View")
    roi_placeholder = st.empty()
    st.info("Ensure adequate lighting for accurate results.")

# Main Dashboard Title
st.markdown("## Live Vitals Monitor")

# Top Row: Metrics
col1, col2, col3 = st.columns(3)
with col1:
    bpm_metric = st.empty()
    bpm_metric.metric("Heart Rate", "--", "BPM")
with col2:
    rr_metric = st.empty()
    rr_metric.metric("Respiration Rate", "--", "BrPM")
with col3:
    status_metric = st.empty()
    status_metric.warning("System Standby")

# Main Content Row: Camera (Left) and Graph (Right)
# Ratio [2, 1] means camera is twice as wide as graph
col_main_cam, col_main_graph = st.columns([2, 1])

with col_main_cam:
    st.caption("Live Camera Feed")
    camera_placeholder = st.empty()

with col_main_graph:
    st.caption("Real-time Signal")
    chart_placeholder = st.empty()

# ===============================
# 3. LOGIC INITIALIZATION
# ===============================
FOREHEAD = [10, 338, 297]
fs = 30  # camera FPS

# Initialize session state (Your original Logic)
if "green_signal" not in st.session_state:
    st.session_state.green_signal = []
if "bpm_list" not in st.session_state:
    st.session_state.bpm_list = []

# ===============================
# 4. MAIN LOOP
# ===============================
if run_monitoring:
    cap = get_camera()
    status_metric.info("Calibrating...")
    
    while True:
        # Check toggle to stop safely
        if not run_monitoring:
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            forehead = extract_roi(frame, landmarks, FOREHEAD)

            if forehead.size > 0:
                # --- EXACT ORIGINAL LOGIC START ---
                st.session_state.green_signal.append(np.mean(forehead[:, :, 1]))

                # Filter and BPM calculation
                if len(st.session_state.green_signal) >= 150:
                    status_metric.success("Monitoring Active")
                    
                    filtered_signal = bandpass(st.session_state.green_signal, fs)
                    bpm = calculate_bpm(filtered_signal, fs)
                    
                    st.session_state.bpm_list.append(bpm)
                    if len(st.session_state.bpm_list) > 5:
                        st.session_state.bpm_list.pop(0)
                    
                    bpm_smooth = int(np.mean(st.session_state.bpm_list))
                    
                    # Update UI Metric
                    bpm_metric.metric("Heart Rate", f"{bpm_smooth}", "BPM")

                # Respiration Rate
                if len(st.session_state.green_signal) >= 300:
                    rr = calculate_respiration(st.session_state.green_signal, fs)
                    
                    # Update UI Metric
                    rr_metric.metric("Respiration Rate", f"{int(rr)}", "BrPM")
                
                # Show ROI in Sidebar (Debug)
                roi_placeholder.image(forehead, channels="BGR", width=150)
                
                # Update Signal Graph (Beside Camera)
                if len(st.session_state.green_signal) > 2:
                    # Show last 100 points
                    chart_data = st.session_state.green_signal[-100:] 
                    # Note: height=350 makes it roughly same height as standard webcam view
                    chart_placeholder.line_chart(chart_data, height=350)
                # --- EXACT ORIGINAL LOGIC END ---

        # Show camera frame
        camera_placeholder.image(frame, channels="BGR")

    cap.release()
else:
    status_metric.warning("Paused")
import tempfile
import cv2
import streamlit as st

from detector import Detector
from tracker import SimpleTracker
from analyzer import Analyzer
from utils import draw_boxes, draw_dashboard, get_roi_box, draw_roi
from settings import (
    TRACK_MAX_DISTANCE,
    TRACK_MAX_LOST,
    ROI_TOP_RATIO,
    ROI_BOTTOM_RATIO,
    ROI_LEFT_RATIO,
    ROI_RIGHT_RATIO,
)

st.set_page_config(page_title="Smart Traffic Analyzer", layout="wide")

st.title("🚦 Smart Traffic Analyzer")
st.write("Upload a road video to detect congestion, wrong-way driving, and possible accident conditions.")
st.write("Analyzer version: v3")

detector = Detector()
tracker = SimpleTracker(max_distance=TRACK_MAX_DISTANCE, max_lost=TRACK_MAX_LOST)
analyzer = Analyzer()

uploaded_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"Uploaded file size: {file_size_mb:.1f} MB")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    stframe = st.empty()
    info_box = st.empty()
    progress_bar = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    skip_frames = max(1, fps // 5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_index += 1
        if frame_index % skip_frames != 0:
            continue

        roi_box = get_roi_box(
            frame.shape,
            ROI_TOP_RATIO,
            ROI_BOTTOM_RATIO,
            ROI_LEFT_RATIO,
            ROI_RIGHT_RATIO,
        )

        detections = detector.detect(frame, roi_box=roi_box)
        objects = tracker.update(detections)
        metrics = analyzer.analyze(objects)

        draw_roi(frame, roi_box)
        draw_boxes(frame, objects)
        frame = draw_dashboard(frame, metrics)

        stframe.image(frame, channels="BGR", use_container_width=True)

        info_box.markdown(
            f"""
## Live Metrics
- **Status:** {metrics['status']}
- **Vehicles:** {metrics['vehicle_count']}
- **Slow Vehicles:** {metrics['slow_count']}
- **Wrong Way:** {metrics['wrong_way_count']}
- **Average Motion:** {metrics['avg_motion']}
"""
        )

        if total_frames > 0:
            progress_bar.progress(min(frame_index / total_frames, 1.0))

    cap.release()

    summary = analyzer.final_summary()

    st.success("Video analysis completed.")
    st.markdown("## Final Summary")
    st.markdown(f"- **Final Status:** {summary['final_status']}")
    st.markdown(f"- **Status Counts:** {summary['status_counts']}")

st.markdown("---")
st.subheader("💬 Traffic Assistant Chatbot")

with st.chat_message("assistant"):
    st.write("Hello! I'm your traffic analysis assistant. Ask me anything about the video analysis!")

user_question = st.chat_input("Ask me about traffic analysis...")

if user_question:
    st.chat_message("user").write(user_question)
    
    response = "I'm here to help with traffic analysis. You can ask me about congestion, accidents, or vehicle counts!"
    
    if "congestion" in user_question.lower():
        response = "🚦 Congestion happens when slow vehicles exceed 5 out of at least 8 vehicles."
    elif "accident" in user_question.lower():
        response = "⚠️ Possible accident is detected when slow vehicles exceed 8 and average motion is below 2.0."
    elif "wrong way" in user_question.lower():
        response = "🔄 Wrong way driving is detected based on direction deviation from the majority."
    elif "vehicle" in user_question.lower() or "car" in user_question.lower():
        response = "🚗 Vehicle count is displayed live in the metrics panel."
    elif "speed" in user_question.lower():
        response = "🏎️ Slow speed threshold is set to 3.0 motion units."
    elif "help" in user_question.lower():
        response = "I can answer questions about congestion, accidents, wrong way driving, vehicles, and speed."
    
    st.chat_message("assistant").write(response)

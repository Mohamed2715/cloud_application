import tempfile
import cv2
import streamlit as st
import imageio
import os

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

    stframe = st.empty()
    info_box = st.empty()
    progress_bar = st.progress(0)

    reader = imageio.get_reader(video_path, 'ffmpeg')
    total_frames = reader.count_frames()
    frame_index = 0

    for frame in reader:
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

        stframe.image(frame, channels="BGR", use_column_width=True)

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

        frame_index += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_index / total_frames, 1.0))

    reader.close()

    summary = analyzer.final_summary()

    st.success("Video analysis completed.")
    st.markdown("## Final Summary")
    st.markdown(f"- **Final Status:** {summary['final_status']}")
    st.markdown(f"- **Status Counts:** {summary['status_counts']}")

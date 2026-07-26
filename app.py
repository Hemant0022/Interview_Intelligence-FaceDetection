import streamlit as st
import cv2
import json
import time
import threading
import copy

from detection import process_frame

# --------------------------------------------------
# Display filtering
# --------------------------------------------------
# These fields are computed (and still used internally for scoring/status
# logic) but should never be shown in the UI, per requirements.
HIDDEN_ATTENTION_FIELDS = ("score", "yaw", "pitch", "roll")
HIDDEN_FACE_FIELDS = ("mesh_detected", "landmarks")


def filtered_view(analysis):
    """Returns a copy of analysis with the hidden fields removed, for display only."""
    view = copy.deepcopy(analysis)
    for field in HIDDEN_ATTENTION_FIELDS:
        view["attention"].pop(field, None)
    for field in HIDDEN_FACE_FIELDS:
        view["face"].pop(field, None)
    return view

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Interview Intelligence",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Interview Intelligence System")
st.caption("AI Based Candidate Attention & Distraction Detection")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("System")

camera_status = st.sidebar.empty()
fps_box = st.sidebar.empty()
time_box = st.sidebar.empty()

# --------------------------------------------------
# Layout
# --------------------------------------------------

left, right = st.columns([2,1])

with left:

    frame_placeholder = st.empty()

with right:

    st.subheader("✅ System Verification")
    verification_box = st.empty()

    st.divider()

    st.subheader("📡 Live Monitoring")
    monitoring_box = st.empty()

st.divider()

st.subheader("Live JSON Output")

json_box = st.empty()

# --------------------------------------------------
# Threaded webcam reader
# --------------------------------------------------
# cv2.VideoCapture keeps an internal driver-side frame buffer. If our
# processing (YOLO + FaceMesh + solvePnP) takes longer than the camera's
# native frame interval, unread frames pile up in that buffer and every
# cap.read() hands back an increasingly STALE frame — the visible lag
# keeps growing the longer the app runs, even if FPS looks fine.
#
# Fix: read frames continuously in a background thread as fast as the
# camera provides them, always keeping only the MOST RECENT frame. The
# main loop below then always processes the latest frame available,
# never a backlog, so lag can't accumulate (frames are dropped instead).

class LatestFrameCamera:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        # Ask the camera for a smaller native resolution. Every frame at
        # 1080p/720p costs extra time in color conversion, inference, and
        # browser image encoding for no benefit if you only display it at
        # a fraction of that size anyway. Not all cameras honor this
        # exactly, but most will get close.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass  # not all backends support this
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


cap = LatestFrameCamera(0)

camera_status.success("Camera Connected")

prev = time.perf_counter()

# --------------------------------------------------
# Frame-skip / UI update throttling
# --------------------------------------------------
# PROCESS_EVERY_N_FRAMES controls how many CAPTURED frames we actually
# run through process_frame() (YOLO + FaceMesh + solvePnP). Frames that
# are skipped just reuse the last known analysis — this is on top of the
# internal YOLO/YuNet throttling inside detection.py, and is the main
# lever for cutting delay further: raise it if it's still too slow.
#
# The VIDEO/PANEL/JSON constants below only control how often we PUSH
# already-computed results to the browser (a separate, cheaper knob).

PROCESS_EVERY_N_FRAMES = 3          # run full inference on every 3rd captured frame
VIDEO_UPDATE_EVERY_N_FRAMES = 2     # redraw the video feed every 2nd frame
PANEL_UPDATE_EVERY_N_FRAMES = 5     # redraw the text panels every 5th frame
JSON_UPDATE_INTERVAL_SECONDS = 1.0  # redraw the JSON blob at most once/sec

frame_count = 0
last_json_update = 0.0
last_analysis = None

# --------------------------------------------------
# Loop
# --------------------------------------------------

try:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret or frame is None:

            # During the first few milliseconds the background thread may
            # not have delivered a frame yet — that's not a camera error.
            time.sleep(0.01)
            continue

        frame_count += 1

        # ----------------------------------------------
        # Skip inference on some frames to cut delay further;
        # reuse the last known analysis for the skipped ones.
        # ----------------------------------------------

        if frame_count % PROCESS_EVERY_N_FRAMES == 0 or last_analysis is None:
            frame, analysis = process_frame(frame)
            last_analysis = analysis
        else:
            analysis = last_analysis

        # ----------------------------------------------

        current = time.perf_counter()

        elapsed = current - prev

        fps = 1 / elapsed if elapsed > 0 else 0.0

        prev = current

        analysis["system"]["fps"] = round(fps,2)

        # ----------------------------------------------
        # Video feed (throttled)
        # ----------------------------------------------

        if frame_count % VIDEO_UPDATE_EVERY_N_FRAMES == 0:

            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                rgb,
                channels="RGB",
                use_container_width=True
            )

        # ----------------------------------------------
        # Sidebar (cheap, keep every frame)
        # ----------------------------------------------

        fps_box.metric(
            "FPS",
            round(fps,2)
        )

        time_box.write(
            analysis["timestamp"]
        )

        # ----------------------------------------------
        # Structured panels (throttled)
        # ----------------------------------------------

        if frame_count % PANEL_UPDATE_EVERY_N_FRAMES == 0:

            face_ok = analysis['face']['detected']
            lighting_ok = analysis['quality']['lighting'] == "Good"
            blur_ok = analysis['quality']['blur'] == "Sharp"
            visibility_ok = analysis['face']['visibility'] == "Full"
            size_ok = analysis['face']['size'] == "Good"

            # ---- System Verification: face + quality checks only ----
            verification_box.markdown(f"""
| Check | Status |
|---|---|
| Face Detected | {'✅' if face_ok else '❌'} {face_ok} |
| Visibility | {'✅' if visibility_ok else '⚠️'} {analysis['face']['visibility']} |
| Size | {'✅' if size_ok else '⚠️'} {analysis['face']['size']} |
| Lighting | {'✅' if lighting_ok else '⚠️'} {analysis['quality']['lighting']} |
| Brightness | {analysis['quality']['brightness']} |
| Blur | {'✅' if blur_ok else '⚠️'} {analysis['quality']['blur']} |
""")

            # ---- Live Monitoring: everything else except the hidden fields ----
            monitoring_box.markdown(f"""
#### {'🟢' if analysis['attention']['status']=="Focused" else '🔴'} {analysis['attention']['status']}

**Attention**

| Parameter | Value |
|-----------|-------|
| Looking at Screen | {analysis['attention']['looking_at_screen']} |
| Head Direction | {analysis['attention']['head_direction']} |
| Gaze Direction | {analysis['attention']['gaze_direction']} |

**Behavior**

| Parameter | Value |
|-----------|-------|
| Looking Left | {analysis['behavior']['looking_left']} |
| Looking Right | {analysis['behavior']['looking_right']} |
| Looking Down | {analysis['behavior']['looking_down']} |
| Multiple Faces | {analysis['behavior']['multiple_faces']} |
| Face Missing | {analysis['behavior']['face_missing']} |

**Objects**

| Parameter | Value |
|-----------|-------|
| Phone | {analysis['objects']['phone_detected']} |
| Person Count | {analysis['objects']['person_count']} |
| Object On Face | {analysis['objects']['object_on_face']} |
| Object On Eyes | {analysis['objects']['object_on_eyes']} |
""")

        # ----------------------------------------------
        # JSON (throttled by time, not frame count, since
        # this is the most expensive widget to re-render).
        # Shows the same filtered view as the panels above —
        # score/yaw/pitch/roll/mesh_detected/landmarks are
        # still computed internally but never displayed.
        # ----------------------------------------------

        if current - last_json_update >= JSON_UPDATE_INTERVAL_SECONDS:

            json_box.json(filtered_view(analysis))

            last_json_update = current

finally:

    # Guarantees the camera (and its background thread) is released even
    # if an exception interrupts the loop — otherwise the OS keeps showing
    # the camera as "in use" while the Streamlit script sits on an error.
    cap.release()
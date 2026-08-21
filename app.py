import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime, timedelta
import pandas as pd

# Import custom modules
import utils
import database
import detector
import ocr
from config import CSV_PATH

# Page configuration
st.set_page_config(
    page_title="AI Multi-Stream Vehicle Tracking & Path Tracing System",
    page_icon="🚗",
    layout="wide"
)

# Custom dark-theme premium styles for visual excellence
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Header */
.main-title {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #3b82f6 100%);
    padding: 28px;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-title h1 {
    font-size: 2.3rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
}

.main-title p {
    font-size: 1.05rem;
    margin: 8px 0 0 0;
    color: #bfdbfe;
    font-weight: 300;
}

/* Premium Card Panels */
.metric-card {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    text-align: center;
    transition: transform 0.2s ease-in-out;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(59, 130, 246, 0.5);
}

.metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.8px;
}

.metric-value {
    color: #3b82f6;
    font-size: 2.2rem;
    font-weight: 700;
    margin-top: 5px;
}

/* CCTV grid styles */
.cctv-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

.cctv-header {
    font-weight: 600;
    font-size: 1rem;
    color: #f1f5f9;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
}

.cctv-meta {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Status logs */
.detection-log-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px;
    font-family: monospace;
    font-size: 0.9rem;
    color: #38bdf8;
    max-height: 300px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# Initialize systems
utils.ensure_directories()
database.initialize_db()

# Load models safely (cached so they load once)
@st.cache_resource
def load_models():
    """
    Loads YOLO vehicle/plate detector and EasyOCR reader.
    """
    det = detector.LicensePlateDetector()
    reader = ocr.LicensePlateOCR(use_gpu=False)
    return det, reader

try:
    with st.spinner("Initializing AI Models (YOLOv8 Multi-Stage Tracking & EasyOCR)... Please wait."):
        det_model, ocr_model = load_models()
except Exception as e:
    st.error(f"Error loading AI Models: {e}")
    st.stop()

# Header banner
st.markdown("""
<div class="main-title">
    <h1>AI-Based Parallel Vehicle Tracking & Spatial-Temporal Path Retrieval System</h1>
    <p>College Final Year Project - Fully Optimized 8-Box CCTV Simulation & Devanagari Localization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.image("https://img.icons8.com/clouds/200/000000/car-roof-box.png", width=120)
st.sidebar.header("📥 Global Parameters")

# Sidebar settings
frame_skip = st.sidebar.slider("⚙️ Frame Skipping (Optimization)", min_value=1, max_value=30, value=10, 
                            help="Process every Nth frame to accelerate parallel speeds. Higher = Faster UI.")
conf_thresh = st.sidebar.slider("🎯 YOLO Detection Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

# Thread Lock for sequential inference calls across threads to prevent crashes/contention
inference_lock = threading.Lock()

# Persistent state manager that survives Streamlit code re-evaluations
class StreamManager:
    shared_status = {
        i: {
            "status": "Idle",
            "location": f"Camera {i+1}",
            "start_time": "",
            "current_frame": 0,
            "total_frames": 0,
            "fps": 0.0,
            "progress": 0.0,
            "latest_plate": "N/A",
            "detections_count": 0,
            "latest_frame": None,
        } for i in range(8)
    }
    global_log = []
    log_lock = threading.Lock()
    active_threads = []
    processing_active = False

# Instantiate or keep references to Manager
if 'manager' not in st.session_state:
    st.session_state.manager = StreamManager()

manager = st.session_state.manager

# Main processing thread target
def process_video_thread(stream_idx, video_path, location, start_time_str, frame_skip_rate, threshold):
    # Setup initial values
    manager.shared_status[stream_idx]["status"] = "Processing"
    manager.shared_status[stream_idx]["location"] = location
    manager.shared_status[stream_idx]["start_time"] = start_time_str
    manager.shared_status[stream_idx]["current_frame"] = 0
    manager.shared_status[stream_idx]["progress"] = 0.0
    manager.shared_status[stream_idx]["fps"] = 0.0
    manager.shared_status[stream_idx]["latest_plate"] = "N/A"
    manager.shared_status[stream_idx]["detections_count"] = 0
    manager.shared_status[stream_idx]["latest_frame"] = None
    
    # Register video in DB
    video_id = database.add_video(os.path.basename(video_path), location, start_time_str)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        manager.shared_status[stream_idx]["status"] = "Error Loading Stream"
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    manager.shared_status[stream_idx]["total_frames"] = total_frames
    
    frame_idx = 0
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    
    best_detections = {} # vehicle_id -> best detection info
    
    # Create an idle/unannotated frame bytes as fallback
    ret, initial_frame = cap.read()
    if ret:
        _, jpeg_initial = cv2.imencode('.jpg', initial_frame)
        manager.shared_status[stream_idx]["latest_frame"] = jpeg_initial.tobytes()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened() and manager.processing_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        manager.shared_status[stream_idx]["current_frame"] = frame_idx
        manager.shared_status[stream_idx]["progress"] = min(1.0, frame_idx / total_frames)
        
        # FPS Calculation
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()
            manager.shared_status[stream_idx]["fps"] = current_fps
            
        if frame_idx % frame_skip_rate == 0 or frame_idx == 1:
            annotated_frame = frame.copy()
            
            # Run heavy inferences thread-safely
            with inference_lock:
                results = det_model.detect_and_track(frame, tracker="bytetrack.yaml")
                
                if results and results.boxes:
                    for box in results.boxes:
                        if box.id is None:
                            continue
                        track_id = int(box.id[0])
                        class_id = int(box.cls[0])
                        class_name = det_model.vehicle_model.names[class_id]
                        
                        vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                        vx1 = max(0, vx1)
                        vy1 = max(0, vy1)
                        vx2 = min(frame.shape[1], vx2)
                        vy2 = min(frame.shape[0], vy2)
                        
                        # Draw vehicle box
                        cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 220, 0), 2)
                        cv2.putText(
                            annotated_frame, f"ID {track_id} {class_name}", 
                            (vx1, max(25, vy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 255), 2
                        )
                        
                        # Crop vehicle
                        vehicle_crop = frame[vy1:vy2, vx1:vx2]
                        if vehicle_crop.size == 0:
                            continue
                            
                        # Detect license plate inside vehicle crop
                        plate_results = det_model.plate_model.predict(
                            vehicle_crop, conf=threshold * 0.7, verbose=False
                        )
                        
                        for p_res in plate_results:
                            for p_box in p_res.boxes:
                                px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                                
                                abs_px1 = vx1 + px1
                                abs_py1 = vy1 + py1
                                abs_px2 = vx1 + px2
                                abs_py2 = vy1 + py2
                                
                                # Draw plate box
                                cv2.rectangle(annotated_frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 120, 0), 2)
                                
                                plate_crop = vehicle_crop[py1:py2, px1:px2]
                                if plate_crop.size == 0:
                                    continue
                                    
                                # Run OCR
                                plate_text, ocr_conf = ocr_model.extract_text(plate_crop)
                                
                                if plate_text and len(plate_text) >= 5:
                                    plate_dev = ocr.convert_to_devnagari(plate_text)
                                    current_sec = frame_idx / fps
                                    timestamp_str = utils.format_timestamp(current_sec)
                                    
                                    # Write to CSV
                                    utils.save_detection_to_csv(
                                        timestamp_str, os.path.basename(video_path), 
                                        location, track_id, [abs_px1, abs_py1, abs_px2, abs_py2],
                                        plate_text, plate_dev, ocr_conf
                                    )
                                    
                                    # Label on frame
                                    cv2.putText(
                                        annotated_frame, f"{plate_text}", 
                                        (abs_px1, max(20, abs_py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.55, (255, 120, 0), 2
                                    )
                                    
                                    # Keep track of highest confidence detection
                                    if track_id not in best_detections or ocr_conf > best_detections[track_id]["confidence"]:
                                        crop_name = f"crop_stream_{stream_idx}_veh_{track_id}.jpg"
                                        crop_path = os.path.join("static", "crops", crop_name)
                                        cv2.imwrite(crop_path, plate_crop)
                                        
                                        best_detections[track_id] = {
                                            "plate_english": plate_text,
                                            "plate_devnagari": plate_dev,
                                            "timestamp": timestamp_str,
                                            "confidence": ocr_conf,
                                            "crop_path": crop_path
                                        }
                                        
                                        manager.shared_status[stream_idx]["latest_plate"] = f"{plate_text} ({plate_dev})"
                                        manager.shared_status[stream_idx]["detections_count"] = len(best_detections)
                                        
                                        with manager.log_lock:
                                            manager.global_log.append({
                                                "time": datetime.now().strftime("%H:%M:%S"),
                                                "location": location,
                                                "id": track_id,
                                                "plate": f"{plate_text} / {plate_dev}",
                                                "conf": f"{ocr_conf:.2f}"
                                            })
            
            # Encode frame for UI
            _, jpeg_frame = cv2.imencode('.jpg', annotated_frame)
            manager.shared_status[stream_idx]["latest_frame"] = jpeg_frame.tobytes()
            
    cap.release()
    
    # Save best detections to database
    for track_id, det in best_detections.items():
        database.add_detection_with_devnagari(
            video_id=video_id,
            vehicle_id=track_id,
            plate_number_english=det["plate_english"],
            plate_number_devnagari=det["plate_devnagari"],
            timestamp=det["timestamp"],
            confidence=det["confidence"],
            crop_path=det["crop_path"]
        )
        
    manager.shared_status[stream_idx]["status"] = "Completed"

# Tabs
tab_process, tab_search, tab_analytics, tab_viva = st.tabs([
    "⚙️ Processing & Detection Dashboard", 
    "🔍 Search & Trace Report", 
    "📊 System Analytics",
    "📘 System Viva Guide"
])

# Layout: Processing
with tab_process:
    st.subheader("🖥️ Upload and Process Multiple Video Streams Parallelly")
    
    # Helper Button
    if st.button("🎁 Extract Sample Videos", help="Downloads standard tracking clips to test."):
        with st.spinner("Downloading sample assets..."):
            import generate_test_data
            generate_test_data.download_sample_data()
            st.success("Sample video downloaded successfully as 'static/samples/sample_video.mp4'.")
            
    st.markdown("---")
    st.write("Configure details for up to **8 input video feeds**:")
    
    # Video Input Config Cards Grid
    inputs_col1, inputs_col2, inputs_col3, inputs_col4 = st.columns(4)
    inputs_col5, inputs_col6, inputs_col7, inputs_col8 = st.columns(4)
    
    slots = [inputs_col1, inputs_col2, inputs_col3, inputs_col4, inputs_col5, inputs_col6, inputs_col7, inputs_col8]
    uploaded_files = [None] * 8
    locations = [""] * 8
    start_dates = [None] * 8
    start_times = [None] * 8
    
    for i in range(8):
        with slots[i]:
            st.markdown(f"#### 📹 Feed {i+1}")
            uploaded_files[i] = st.file_uploader(f"Upload Video", type=["mp4", "avi", "mov"], key=f"file_{i}")
            locations[i] = st.text_input(f"Location", value=f"Kathmandu Gate {chr(65+i)}", key=f"loc_{i}")
            
            # Date/Time selector
            col_d, col_t = st.columns(2)
            with col_d:
                start_dates[i] = st.date_input(f"Date", value=datetime.now(), key=f"date_{i}")
            with col_t:
                # Deducting 5-minute increments for realistic tracing simulation
                default_time = (datetime.now() + timedelta(minutes=5*i)).time()
                start_times[i] = st.time_input(f"Time", value=default_time, key=f"time_{i}")
                
            st.markdown("---")

    # Command buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_parallel = st.button("🚀 Start Parallel Video Processing", use_container_width=True)
    with col_btn2:
        stop_parallel = st.button("🛑 Stop Processing", use_container_width=True)
        
    if stop_parallel:
        manager.processing_active = False
        st.warning("Halting all processing streams...")
        
    st.markdown("### 📽️ Real-Time 8-Box CCTV Feed Monitor")
    
    # Set up 4x2 Grid of CCTV Stream boxes
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)
    cctv_cols = [c1, c2, c3, c4, c5, c6, c7, c8]
    
    img_placeholders = [None] * 8
    bar_placeholders = [None] * 8
    stat_placeholders = [None] * 8
    
    # Create grey placeholder image
    grey_placeholder_img = np.zeros((225, 300, 3), dtype=np.uint8) + 40
    # Add text to idle frame
    cv2.putText(
        grey_placeholder_img, "Camera Idle", (65, 120), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2
    )
    _, grey_bytes = cv2.imencode('.jpg', grey_placeholder_img)
    grey_frame = grey_bytes.tobytes()
    
    for i in range(8):
        with cctv_cols[i]:
            st.markdown(f"""
            <div class="cctv-card">
                <div class="cctv-header">
                    <span>📍 {locations[i] or f'Camera {i+1}'}</span>
                    <span id="badge_{i}" style="color: #94a3b8; font-size: 0.8rem;">● Idle</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            img_placeholders[i] = st.empty()
            bar_placeholders[i] = st.empty()
            stat_placeholders[i] = st.empty()
            
            # Show default idle state
            img_placeholders[i].image(grey_frame, use_container_width=True)
            bar_placeholders[i].progress(0.0)
            stat_placeholders[i].write("Status: Idle")

    st.markdown("### 📝 Global Real-Time Activity Log")
    st_global_log_table = st.empty()
    
    # Trigger Parallel Execution
    if start_parallel:
        active_indices = []
        temp_paths = []
        
        # Check active feeds
        for i in range(8):
            if uploaded_files[i] is not None:
                active_indices.append(i)
                # Save temp file
                t_path = os.path.join("static", "samples", f"temp_input_{i}.mp4")
                with open(t_path, "wb") as f:
                    f.write(uploaded_files[i].getbuffer())
                temp_paths.append(t_path)
                
        if not active_indices:
            st.warning("Please upload at least one video to process.")
        else:
            # Clear UI logs and CSV
            manager.global_log = []
            utils.initialize_csv()
            manager.processing_active = True
            
            # Spawn Threads
            manager.active_threads = []
            for idx, i in enumerate(active_indices):
                dt_str = f"{start_dates[i]} {start_times[i].strftime('%H:%M:%S')}"
                t = threading.Thread(
                    target=process_video_thread, 
                    args=(i, temp_paths[idx], locations[i], dt_str, frame_skip, conf_thresh),
                    daemon=True
                )
                manager.active_threads.append(t)
                t.start()
                
            st.info(f"Started parallel streams on {len(active_indices)} threads. Processing...")
            
            # UI Render Loop
            while any(t.is_alive() for t in manager.active_threads):
                for i in range(8):
                    status = manager.shared_status[i]
                    loc = locations[i] or f"Camera {i+1}"
                    
                    if i in active_indices:
                        state_label = status["status"]
                        if state_label == "Processing":
                            fps_val = status["fps"]
                            frame_idx = status["current_frame"]
                            tot = status["total_frames"]
                            badge_color = "#3b82f6"
                            progress_val = status["progress"]
                            
                            stat_text = f"**Status**: Processing ({frame_idx}/{tot}) | **FPS**: {fps_val:.1f} | **Latest**: `{status['latest_plate']}`"
                        elif state_label == "Completed":
                            badge_color = "#22c55e"
                            stat_text = f"**Status**: Completed | **Total Tracked**: {status['detections_count']}"
                            progress_val = 1.0
                        else:
                            badge_color = "#ef4444"
                            stat_text = f"**Status**: {state_label}"
                            progress_val = 0.0
                            
                        # Show frame if available
                        if status["latest_frame"] is not None:
                            img_placeholders[i].image(status["latest_frame"], use_container_width=True)
                            
                        bar_placeholders[i].progress(progress_val)
                        stat_placeholders[i].markdown(stat_text)
                    else:
                        # Idle streams
                        img_placeholders[i].image(grey_frame, use_container_width=True)
                        bar_placeholders[i].progress(0.0)
                        stat_placeholders[i].write("Status: Idle")
                
                # Render Global Live log
                with manager.log_lock:
                    if manager.global_log:
                        st_global_log_table.dataframe(pd.DataFrame(manager.global_log[::-1]), use_container_width=True)
                
                time.sleep(0.1) # Sleep to keep UI responsive
                
            # Finish processing updates
            st.success("🎉 Parallel Stream Processing Completed! All records written to CSV and Database.")
            
            # Clean up temp files
            for tp in temp_paths:
                try:
                    os.remove(tp)
                except Exception:
                    pass

# Layout: Search & Trace Report
with tab_search:
    st.subheader("🔍 Space-Insensitive Plate Search and Directed Path Tracer")
    
    search_query = st.text_input(
        "💳 Enter License Plate Number (English or Devanagari)", 
        placeholder="e.g. BA 2 CHA 1234, बा २ च १२३४, or partial..."
    ).strip()
    
    if st.button("🔎 Run Retrieval Search"):
        if not search_query:
            st.warning("Please type a plate number to search.")
        else:
            # Query Database
            results = database.search_plate(search_query)
            
            if not results:
                st.warning("No records matched your search query.")
            else:
                st.success(f"Found {len(results)} matching records in the database:")
                
                # Display Results in a beautiful list with crops
                for row in results:
                    with st.container():
                        col_det1, col_det2, col_det3, col_det4 = st.columns([1, 1, 2, 2])
                        
                        with col_det1:
                            if row['crop_path'] and os.path.exists(row['crop_path']):
                                st.image(row['crop_path'], caption="Best OCR Crop", width=130)
                            else:
                                st.image("https://img.icons8.com/ios/100/000000/license-plate.png", caption="Crop Unavailable", width=100)
                                
                        with col_det2:
                            if row['crop_path'] and os.path.exists(row['crop_path']):
                                raw_crop = cv2.imread(row['crop_path'])
                                prep_crop = utils.preprocess_plate_image(raw_crop)
                                if prep_crop is not None:
                                    st.image(prep_crop, caption="Enhanced (CLAHE)", width=130)
                                    
                        with col_det3:
                            st.markdown(f"#### 💳 English Plate: `{row['plate_number_english']}`")
                            st.markdown(f"#### 🇳🇵 Devanagari Plate: `{row['plate_number_devnagari']}`")
                            st.markdown(f"**📍 Location**: {row['location']}")
                            st.markdown(f"**📅 Video Start Time**: `{row['datetime_record']}`")
                            
                        with col_det4:
                            st.markdown(f"**🎥 Source Video File**: `{row['filename']}`")
                            st.markdown(f"**⏱️ Video Frame Offset**: `{row['timestamp']}`")
                            st.markdown(f"**🎯 OCR Confidence**: `{row['confidence']:.2f}`")
                            st.markdown(f"**🚗 Vehicle ID**: `ID {row['vehicle_id']}`")
                            
                        st.write("---")
                
                # ----------------- PATH TRACING REPORT & GRAPH GENERATION -----------------
                st.markdown("### 🗺️ Vehicle Movement Timeline & Spatial Path Tracer")
                
                # Compute absolute detection times for chronological sorting
                path_data = []
                for row in results:
                    abs_time = database.get_db_connection() # Placeholder
                    # Parse start datetime and add timeline offset
                    abs_time = utils.datetime.strptime(row['datetime_record'], "%Y-%m-%d %H:%M:%S")
                    
                    try:
                        ts_parts = row['timestamp'].split(':')
                        offset_secs = int(ts_parts[0]) * 60 + int(ts_parts[1])
                    except Exception:
                        offset_secs = 0
                        
                    detection_datetime = abs_time + timedelta(seconds=offset_secs)
                    
                    path_data.append({
                        "location": row['location'],
                        "absolute_time": detection_datetime,
                        "video": row['filename'],
                        "plate_eng": row['plate_number_english'],
                        "plate_dev": row['plate_number_devnagari']
                    })
                    
                # Sort path chronologically
                sorted_path = sorted(path_data, key=lambda x: x['absolute_time'])
                
                # Render Timeline Table
                st.markdown("#### Chronological Route Log")
                timeline_list = []
                for idx, pt in enumerate(sorted_path):
                    time_str = pt['absolute_time'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Calculate duration since last seen
                    if idx == 0:
                        elapsed_str = "Start of Trace"
                    else:
                        diff = pt['absolute_time'] - sorted_path[idx-1]['absolute_time']
                        tot_secs = int(diff.total_seconds())
                        mins = tot_secs // 60
                        secs = tot_secs % 60
                        elapsed_str = f"+{mins}m {secs}s"
                        
                    timeline_list.append({
                        "Step": idx + 1,
                        "Spotting Time": time_str,
                        "📍 Spotting Location": pt['location'],
                        "Time From Last Spot": elapsed_str,
                        "Source File": pt['video']
                    })
                st.table(timeline_list)
                
                # Generate and Display network graph path image
                st.markdown("#### Chronological Spatial directed Graph Path Map")
                with st.spinner("Generating vehicle route path graph..."):
                    best_plate_lbl = results[0]['plate_number_english']
                    graph_png = utils.draw_vehicle_path(sorted_path, best_plate_lbl)
                    
                    if graph_png and os.path.exists(graph_png):
                        st.image(graph_png, use_container_width=True)
                        st.info(f"💾 Directed graph image saved to `./graph/trace_{best_plate_lbl.replace(' ', '_')}.png`")
                    else:
                        st.warning("Failed to render route path graph.")

# Layout: System Analytics
with tab_analytics:
    st.subheader("📊 Database Analytics Dashboard")
    
    # Query all records
    all_rows = database.get_all_records()
    
    if not all_rows:
        st.warning("No records in the database. Run video processing first.")
    else:
        df = pd.DataFrame(all_rows)
        
        # Metrics Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📺 Total Videos Scanned</div>
                <div class="metric-value">{len(df['filename'].unique())}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📍 Unique Locations Covered</div>
                <div class="metric-value">{len(df['location'].unique())}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚗 Unique Vehicle Spotting Logs</div>
                <div class="metric-value">{len(df)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### 📍 Detections Count by Location")
            loc_counts = df['location'].value_counts()
            st.bar_chart(loc_counts)
            
        with col_c2:
            st.markdown("#### ⏱️ Hourly Activity Heat Timeline")
            # Extract hours from start record timestamps
            hours_list = []
            for val in df['datetime_record']:
                try:
                    dt_val = datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                    hours_list.append(f"{dt_val.hour:02d}:00")
                except Exception:
                    hours_list.append("Unknown")
            hour_counts = pd.Series(hours_list).value_counts().sort_index()
            st.line_chart(hour_counts)
            
        # Display Database Grid
        st.markdown("#### 🗄️ Full SQLite Database Detections Record Log")
        st.dataframe(df[[
            'id', 'vehicle_id', 'plate_number_english', 'plate_number_devnagari', 
            'location', 'timestamp', 'confidence', 'datetime_record', 'filename'
        ]], use_container_width=True)
        
        # Download scan records CSV button
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
                st.download_button(
                    label="📥 Download Session Raw CSV Scan Record",
                    data=f.read(),
                    file_name="scan_records.csv",
                    mime="text/csv"
                )

# Layout: Viva guide
with tab_viva:
    st.subheader("📘 Project Defense Architecture & College Viva Guide")
    
    col_v1, col_v2 = st.columns([2, 3])
    
    with col_v1:
        st.markdown("### 🏗️ Upgraded System Pipeline")
        st.code("""
  +-------------------------------------------------+
  |               8 CCTV INPUT FEEDS                |
  |        (Video Streams + Locations + Times)       |
  +------------------------+------------------------+
                           |
                           v  (Parallel Background Threads)
  +-------------------------------------------------+
  |         STAGE 1: VEHICLE TRACK & IDENTIFY       |
  |   (YOLOv8 + ByteTrack Object IDs per Stream)    |
  +------------------------+------------------------+
                           |
                           v  (Crop Vehicle & Run Plate Detect)
  +-------------------------------------------------+
  |        STAGE 2: LICENSE PLATE ENHANCE           |
  |      (YOLOv8 Plate model inside Vehicle Crop)   |
  |       (Adaptive CLAHE Contrast Enhancement)     |
  +------------------------+------------------------+
                           |
                           v
  +-------------------------------------------------+
  |            STAGE 3: TEXT OCR & TRANSLATE        |
  |     (EasyOCR Read + English Nepal Correction)    |
  |     (Devanagari Transliteration Mapping Layer)  |
  +------------------------+------------------------+
                           |
                           v  (Thread-Locked Loggers)
  +------------------------+------------------------+
  |  CSV FILE: RAW SPOTS   |  DB: BEST UNIQUE CROP  |
  |   (scan_records.csv)   |  (Highest Confidence)  |
  +------------------------+------------------------+
        """, language="text")
        
    with col_v2:
        st.markdown("### 🤖 Core Enhancements Implemented")
        st.write("""
        *   **8-Stream Parallelization**: Uses background Python threads to read and process 8 distinct CCTV video feeds concurrently.
        *   **Multi-Class ByteTrack Tracking**: Leverages state-of-the-art YOLOv8 ByteTrack to identify and assign persistent tracking IDs to vehicles. This allows frame-to-frame vehicle correlation.
        *   **In-Place UI Updates**: Implements modern Streamlit placeholders to show live video feeds and processing progress without reloading or lagging.
        *   **Double Script (Devanagari/English) DB Logging**: The database logs both English and transliterated Devanagari characters (e.g., `BA 2 CHA 1234` to `बा २ च १२३४`) to satisfy national language embossed requirements in Nepal.
        *   **Chronological Graph Path Tracer**: Traces a vehicle's movements chronologically. When a user inputs a number plate, it queries SQLite, calculates absolute timestamps, sorts them, computes time offsets, and draws a directed graph pathway using `matplotlib` and `networkx` saved in `./graph/`.
        """)
        
    st.markdown("### 🎓 Expected College Viva Q&A")
    
    qa_list = [
        {
            "q": "How does frame skipping optimize performance in your project?",
            "a": "Processing every frame of 8 simultaneous HD videos (30 FPS each) requires massive GPU compute. By implementing frame skipping (e.g. process every 10th frame), we reduce the inference load by 90% (processing only 3 frames per second instead of 30). This allows the entire pipeline to run smoothly on standard laptops or CPU-only setups without dropping the dashboard's responsiveness."
        },
        {
            "q": "Why do you use thread locking (Lock) in your parallel database and CSV writing?",
            "a": "SQLite and file systems are not naturally concurrent-write safe. When 8 background threads process videos in parallel, they may try to write plate detections to the same SQLite database file and CSV file at the exact same millisecond. This leads to write collisions, file lock errors, or database corruption. We implement thread locks (`db_lock` and `csv_lock`) to serialize file and DB access, ensuring that only one thread writes to a resource at any time while keeping the process thread-safe."
        },
        {
            "q": "How does your system construct a spatial-temporal trajectory map?",
            "a": "When a vehicle is spotted, we log the spotting location (e.g. 'Kathmandu Gate A') and absolute time (video start time + timestamp offset of the frame). When a user searches for a plate, we retrieve all spots, sort them chronologically, and calculate the elapsed time between spots. We then construct a directed network graph using networkx where locations are nodes and the chronological timeline determines the arrows, visualizing the path taken by the vehicle."
        }
    ]
    
    for idx, qa in enumerate(qa_list):
        st.markdown(f"**Q{idx+1}: {qa['q']}**")
        st.info(f"**Answer:** {qa['a']}")

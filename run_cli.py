import os
import cv2
import time
from datetime import datetime, timedelta
import utils
import database
import detector
import ocr
from config import SAMPLES_DIR

def run_test_pipeline():
    print("====================================================")
    print("   AI Vehicle Tracking & Path Tracing CLI Runner    ")
    print("====================================================\n")
    
    # 1. Ensure directories exist
    utils.ensure_directories()
    
    # 2. Check and download sample video if not present
    video_path = os.path.join(SAMPLES_DIR, "sample_video.mp4")
    if not os.path.exists(video_path):
        print("Sample video not found. Downloading sample data...")
        import generate_test_data
        generate_test_data.download_sample_data()
    else:
        print(f"Using existing sample video: {video_path}")
        
    if not os.path.exists(video_path):
        print("[Error] Failed to find or download sample video.")
        return
        
    # 3. Initialize database and CSV
    print("Initializing Database & CSV log...")
    database.initialize_db()
    utils.initialize_csv()
    
    # 4. Load Models
    print("Loading AI Models (YOLO26 & EasyOCR)...")
    det = detector.LicensePlateDetector()
    reader = ocr.LicensePlateOCR(use_gpu=False)
    
    # 5. Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open video file: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    duration = total_frames / fps
    print(f"Video Properties: Resolution {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | FPS: {fps:.2f} | Duration: {duration:.1f}s")
    
    # Register video in DB
    location = "Test Camera A"
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_id = database.add_video(os.path.basename(video_path), location, start_time_str)
    
    frame_idx = 0
    frame_skip = 10  # Process every 10th frame for speed
    best_detections = {} # vehicle_id -> best detection info
    
    print(f"\nProcessing video frames (skipping {frame_skip} frames for optimization)...")
    
    start_proc_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Log progress every 50 frames
        if frame_idx % 50 == 0:
            print(f" -> Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
            
        if frame_idx % frame_skip == 0 or frame_idx == 1:
            # Detect and track vehicles
            results = det.detect_and_track(frame, tracker="bytetrack.yaml")
            
            if results and results.boxes:
                for box in results.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    class_name = det.vehicle_model.names[class_id]
                    
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                    vx1 = max(0, vx1)
                    vy1 = max(0, vy1)
                    vx2 = min(frame.shape[1], vx2)
                    vy2 = min(frame.shape[0], vy2)
                    
                    # Crop vehicle
                    vehicle_crop = frame[vy1:vy2, vx1:vx2]
                    if vehicle_crop.size == 0:
                        continue
                        
                    # Detect license plate inside vehicle crop
                    plate_results = det.plate_model.predict(
                        vehicle_crop, conf=0.25 * 0.7, verbose=False
                    )
                    
                    for p_res in plate_results:
                        for p_box in p_res.boxes:
                            px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                            
                            abs_px1 = vx1 + px1
                            abs_py1 = vy1 + py1
                            abs_px2 = vx1 + px2
                            abs_py2 = vy1 + py2
                            
                            plate_crop = vehicle_crop[py1:py2, px1:px2]
                            if plate_crop.size == 0:
                                continue
                                
                            # OCR
                            plate_text, ocr_conf = reader.extract_text(plate_crop)
                            
                            if plate_text and len(plate_text) >= 5:
                                plate_dev = ocr.convert_to_devnagari(plate_text)
                                timestamp_str = utils.format_timestamp(frame_idx / fps)
                                
                                # Log frame-level detection to CSV
                                utils.save_detection_to_csv(
                                    timestamp_str, os.path.basename(video_path), 
                                    location, track_id, [abs_px1, abs_py1, abs_px2, abs_py2],
                                    plate_text, plate_dev, ocr_conf
                                )
                                
                                print(f"   [Scan] Frame {frame_idx} | ID {track_id} ({class_name}) | Plate (Eng): '{plate_text}' | Plate (Dev): '{plate_dev}' | Conf: {ocr_conf:.2f}")
                                
                                # Update best unique detection
                                if track_id not in best_detections or ocr_conf > best_detections[track_id]["confidence"]:
                                    crop_name = f"crop_cli_veh_{track_id}.jpg"
                                    crop_path = os.path.join("static", "crops", crop_name)
                                    cv2.imwrite(crop_path, plate_crop)
                                    
                                    best_detections[track_id] = {
                                        "plate_english": plate_text,
                                        "plate_devnagari": plate_dev,
                                        "timestamp": timestamp_str,
                                        "confidence": ocr_conf,
                                        "crop_path": crop_path
                                    }
                                    
    cap.release()
    proc_duration = time.time() - start_proc_time
    print(f"\nProcessing finished in {proc_duration:.2f} seconds.")
    print(f"Total vehicle IDs tracked: {len(best_detections)}")
    
    # Save best detections to SQLite
    print("Saving highest confidence vehicle detections to SQLite database...")
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
        print(f" -> Logged ID {track_id}: {det['plate_english']} / {det['plate_devnagari']} (Conf: {det['confidence']:.2f})")
        
    print("\n====================================================")
    print("   Detections Logged in SQLite database:            ")
    print("====================================================")
    all_records = database.get_all_records()
    for rec in all_records:
        print(f"Vehicle ID {rec['vehicle_id']} | English: {rec['plate_number_english']} | Devanagari: {rec['plate_number_devnagari']} | Conf: {rec['confidence']:.2f} | Time: {rec['timestamp']} | Location: {rec['location']}")
        
    # Generate Route Timeline Graph for the first detected plate to demonstrate graph builder
    if best_detections:
        first_veh_id = list(best_detections.keys())[0]
        first_det = best_detections[first_veh_id]
        print(f"\nGenerating route path graph for {first_det['plate_english']}...")
        
        # Construct path trace array
        path_timeline = [{
            "location": location,
            "absolute_time": datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=0),
            "video": os.path.basename(video_path)
        }]
        
        graph_png = utils.draw_vehicle_path(path_timeline, first_det['plate_english'])
        if graph_png and os.path.exists(graph_png):
            print(f"[Success] Directed graph path visualizer saved to: {graph_png}")
            
    print("\nCSV scan records saved to: static/scan_records.csv")
    print("Database file updated at: database/vehicle_records.db")
    print("\nRun 'streamlit run app.py' to view the interactive dashboard!")
    print("====================================================")

if __name__ == "__main__":
    run_test_pipeline()

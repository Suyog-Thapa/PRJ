import os
import urllib.request
from ultralytics import YOLO
import cv2
import numpy as np
from config import PLATE_MODEL_PATH, VEHICLE_MODEL_PATH

# Fallback URLs for downloading the license plate detector
MODEL_URLS = [
    "https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt",
    "https://huggingface.co/AZIIIIIIIIZ/License-plate-detection/resolve/main/best.pt"
]

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate box areas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

class LicensePlateDetector:
    def __init__(self):
        """
        Initializes the two-stage detector.
        Stage 1: YOLOv8 vehicle detection (yolov8n.pt).
        Stage 2: YOLOv8 license plate detection (license_plate_detector.pt).
        """
        self.plate_model = None
        self.vehicle_model = None
        
        # Load plate detector
        self.load_plate_model_safely()
        # Load vehicle detector
        self.load_vehicle_model_safely()

    def download_model(self, url):
        os.makedirs(os.path.dirname(PLATE_MODEL_PATH), exist_ok=True)
        print(f"Downloading pretrained license plate detector from: {url}")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, PLATE_MODEL_PATH)
        print(f"Model saved to: {PLATE_MODEL_PATH}")

    def load_plate_model_safely(self):
        if os.path.exists(PLATE_MODEL_PATH):
            try:
                print("Loading existing YOLOv8 License Plate Detector model...")
                self.plate_model = YOLO(PLATE_MODEL_PATH)
                print("Plate model loaded successfully.")
                return
            except Exception as e:
                print(f"Corrupted plate model detected: {e}. Deleting and re-downloading...")
                try:
                    os.remove(PLATE_MODEL_PATH)
                except Exception:
                    pass

        for url in MODEL_URLS:
            try:
                self.download_model(url)
                print("Loading YOLOv8 License Plate Detector model...")
                self.plate_model = YOLO(PLATE_MODEL_PATH)
                print("Plate model loaded successfully!")
                return
            except Exception as e:
                print(f"Failed to load from {url}: {e}")
                if os.path.exists(PLATE_MODEL_PATH):
                    try:
                        os.remove(PLATE_MODEL_PATH)
                    except Exception:
                        pass
        raise RuntimeError("Failed to load or download license plate model.")

    def load_vehicle_model_safely(self):
        """
        Loads the general vehicle detector model (YOLOv8 Nano).
        """
        try:
            print("Loading YOLOv8 Vehicle Detector model...")
            self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
            print("Vehicle model loaded successfully.")
        except Exception as e:
            print(f"Failed to load vehicle model from local models/ path: {e}")
            print("Loading standard yolov8n.pt (will download automatically if needed)...")
            self.vehicle_model = YOLO("yolov8n.pt")
            print("Vehicle model loaded successfully.")

    def detect_and_track(self, frame, tracker="bytetrack.yaml"):
        """
        Runs YOLOv8 vehicle detection and tracking (ByteTrack).
        Returns the tracked results object.
        """
        if frame is None or frame.size == 0:
            return None
            
        vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
        
        try:
            results = self.vehicle_model.track(
                source=frame,
                persist=True,
                classes=vehicle_classes,
                tracker=tracker,
                verbose=False
            )
            if results and len(results) > 0:
                return results[0]
            return None
        except Exception as e:
            print(f"Tracking error: {e}")
            # Fallback to standard prediction without tracking IDs
            try:
                results = self.vehicle_model.predict(
                    source=frame,
                    classes=vehicle_classes,
                    verbose=False
                )
                if results and len(results) > 0:
                    return results[0]
                return None
            except Exception:
                return None

    def detect(self, frame, confidence_threshold=0.25):
        """
        Runs the Two-Stage detection pipeline (static frame backup):
        1. Detects all vehicles (cars, trucks, motorcycles, buses).
        2. Crops each vehicle and runs the license plate detector inside the crop.
        3. Runs full-frame detection as a fallback.
        4. Merges all detections using IoU non-maximum suppression.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        all_plate_detections = []

        # --- STAGE 1: VEHICLE DETECTION ---
        vehicle_classes = [2, 3, 5, 7]
        try:
            vehicle_results = self.vehicle_model.predict(frame, conf=0.3, classes=vehicle_classes, verbose=False)
            vehicles = []
            for result in vehicle_results:
                for box in result.boxes:
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                    vx1 = max(0, vx1 - 15)
                    vy1 = max(0, vy1 - 15)
                    vx2 = min(w, vx2 + 15)
                    vy2 = min(h, vy2 + 15)
                    vehicles.append((vx1, vy1, vx2, vy2))
        except Exception as e:
            print(f"Vehicle detection failed: {e}. Falling back to single-stage.")
            vehicles = []

        # --- STAGE 2: CROP VEHICLES AND DETECT PLATES ---
        for vx1, vy1, vx2, vy2 in vehicles:
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue
                
            crop_results = self.plate_model.predict(vehicle_crop, conf=confidence_threshold * 0.7, verbose=False)
            for result in crop_results:
                for box in result.boxes:
                    cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    gx1 = vx1 + cx1
                    gy1 = vy1 + cy1
                    gx2 = vx1 + cx2
                    gy2 = vy1 + cy2
                    
                    gx1 = max(0, gx1)
                    gy1 = max(0, gy1)
                    gx2 = min(w, gx2)
                    gy2 = min(h, gy2)
                    
                    all_plate_detections.append({
                        "box": (gx1, gy1, gx2, gy2),
                        "confidence": conf,
                        "crop": frame[gy1:gy2, gx1:gx2]
                    })

        # --- STAGE 3: FULL FRAME FALLBACK ---
        full_results = self.plate_model.predict(frame, conf=confidence_threshold, verbose=False)
        for result in full_results:
            for box in result.boxes:
                fx1, fy1, fx2, fy2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                fx1 = max(0, fx1)
                fy1 = max(0, fy1)
                fx2 = min(w, fx2)
                fy2 = min(h, fy2)
                
                all_plate_detections.append({
                    "box": (fx1, fy1, fx2, fy2),
                    "confidence": conf,
                    "crop": frame[fy1:fy2, fx1:fx2]
                })

        # --- STAGE 4: MERGE DUPLICATES ---
        if not all_plate_detections:
            return []

        all_plate_detections = sorted(all_plate_detections, key=lambda x: x["confidence"], reverse=True)
        merged_detections = []
        
        for det in all_plate_detections:
            box = det["box"]
            keep = True
            for kept_det in merged_detections:
                iou = calculate_iou(box, kept_det["box"])
                if iou > 0.45:
                    keep = False
                    break
            if keep:
                merged_detections.append(det)

        return merged_detections

    def draw_detections(self, frame, detections):
        annotated_frame = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            label = f"Plate: {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_h - 10), 
                (x1 + text_w, y1), 
                (0, 255, 0), 
                cv2.FILLED
            )
            cv2.putText(
                annotated_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2, 
                cv2.LINE_AA
            )
        return annotated_frame

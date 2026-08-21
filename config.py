import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "database")
GRAPH_DIR = os.path.join(BASE_DIR, "graph")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CROPS_DIR = os.path.join(STATIC_DIR, "crops")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

# Models
VEHICLE_MODEL_PATH = os.path.join(MODEL_DIR, "yolo26n.pt")
PLATE_MODEL_PATH = os.path.join(MODEL_DIR, "license_plate_detector.pt")

# Database & CSV paths
DB_PATH = os.path.join(DB_DIR, "vehicle_records.db")
CSV_PATH = os.path.join(STATIC_DIR, "scan_records.csv")

import os
import cv2
import re
import csv
import threading
import matplotlib
matplotlib.use('Agg') # Thread-safe non-GUI backend
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from config import CSV_PATH

csv_lock = threading.Lock()

def ensure_directories():
    """
    Ensures that the required directories for database, models, and crop images exist.
    """
    directories = [
        "models",
        "database",
        "graph",
        os.path.join("static", "crops"),
        os.path.join("static", "samples")
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

def clean_plate_text(text):
    """
    Cleans OCR output by keeping only uppercase alphanumeric characters.
    """
    if not text:
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned.upper()

def preprocess_plate_image(img):
    """
    Applies advanced image enhancement to clear up blurry or dim plates:
    1. Grayscale conversion.
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) to balance lighting/dimness.
    3. Bilateral Filter to reduce noise while keeping characters sharp.
    4. Smart scaling using bicubic interpolation.
    """
    if img is None or img.size == 0:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    h, w = denoised.shape[:2]
    upscaled = cv2.resize(denoised, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    
    return upscaled

def format_timestamp(seconds):
    """
    Formats video duration in seconds to MM:SS format.
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def initialize_csv():
    """
    Initializes/clears the global CSV log file and writes the headers.
    """
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with csv_lock:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp Offset", "Video Name", "Location", "Vehicle ID", 
                "Plate Box (xyxy)", "Plate English", "Plate Devanagari", "Confidence"
            ])
    print(f"CSV initialized at: {CSV_PATH}")

def save_detection_to_csv(timestamp, video_name, location, vehicle_id, plate_box, plate_english, plate_devnagari, confidence):
    """
    Appends a single frame's raw detection to the CSV log. Thread-safe.
    """
    with csv_lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, video_name, location, vehicle_id, 
                str(plate_box), plate_english, plate_devnagari, f"{confidence:.4f}"
            ])

def draw_vehicle_path(detections, plate_number):
    """
    Creates and saves a beautiful directed graph showing the path and timeline of a vehicle.
    Saves the image to `./graph/trace_{plate_number}.png`.
    """
    if not detections:
        return None
        
    os.makedirs("graph", exist_ok=True)
    graph_filename = f"trace_{plate_number.replace(' ', '_')}.png"
    graph_path = os.path.join("graph", graph_filename)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Sort detections by absolute datetime
    sorted_dets = sorted(detections, key=lambda x: x['absolute_time'])
    
    labels = {}
    node_colors = []
    
    for i, det in enumerate(sorted_dets):
        node_id = i
        loc_time_label = f"{det['location']}\n{det['absolute_time'].strftime('%H:%M:%S')}"
        G.add_node(node_id, label=loc_time_label)
        labels[node_id] = loc_time_label
        
        # Color gradient: start is green, end is red, intermediate are blue
        if i == 0:
            node_colors.append('#22c55e') # Green
        elif i == len(sorted_dets) - 1:
            node_colors.append('#ef4444') # Red
        else:
            node_colors.append('#3b82f6') # Blue
            
        if i > 0:
            G.add_edge(i - 1, i)
            
    # Set up matplotlib figure
    plt.figure(figsize=(10, 4), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    
    # Positioning nodes horizontally from left to right to represent time progression
    pos = {}
    for i in range(len(sorted_dets)):
        pos[i] = (i * 2.5, 0) # Constant y-coordinate, linear spacing on x
        
    # Draw graph elements
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=2500, 
        edgecolors='white', 
        linewidths=2,
        ax=ax
    )
    
    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos, 
        edge_color='#94a3b8', 
        width=3, 
        arrowstyle='-|>', 
        arrowsize=20,
        connectionstyle='arc3,rad=0.15',
        ax=ax
    )
    
    # Draw labels with high readability
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels, 
        font_size=8, 
        font_color='white', 
        font_family='sans-serif',
        font_weight='bold',
        ax=ax
    )
    
    plt.title(f"📍 TRACE PATH REPORT: {plate_number}", color='white', fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(graph_path, format='png', dpi=300, facecolor='#0f172a')
    plt.close()
    
    return graph_path

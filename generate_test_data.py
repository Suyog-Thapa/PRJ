import os
import urllib.request
import cv2
import numpy as np

VIDEO_URL = "https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8/raw/main/sample.mp4"
SAMPLE_DIR = os.path.join("static", "samples")
VIDEO_PATH = os.path.join(SAMPLE_DIR, "sample_video.mp4")
IMAGE_PATH = os.path.join(SAMPLE_DIR, "sample_image.jpg")

def create_synthetic_data():
    """
    Generates synthetic vehicle and plate data as an offline fallback.
    """
    print("Generating synthetic offline test data...")
    # 1. Create a synthetic sample image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background (Road/Sky)
    img[:200, :] = [200, 180, 150]  # Light sky/horizon
    img[200:, :] = [80, 80, 80]     # Dark road
    
    # Draw a mock car (Dark Blue Rectangle)
    cv2.rectangle(img, (150, 150), (490, 380), (120, 50, 50), -1)
    # Windshield
    cv2.rectangle(img, (180, 160), (460, 240), (200, 200, 200), -1)
    
    # Draw a license plate (White rectangle with black border)
    plate_x1, plate_y1, plate_x2, plate_y2 = 250, 300, 390, 340
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 255), -1)
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), 2)
    
    # Add text on the plate
    cv2.putText(
        img, "NY-9831", (plate_x1 + 10, plate_y1 + 30), 
        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
    )
    
    # Save the synthetic image
    cv2.imwrite(IMAGE_PATH, img)
    print(f"Saved synthetic sample image to: {IMAGE_PATH}")

    # 2. Create a synthetic video (a moving car with a plate)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_PATH, fourcc, 20.0, (640, 480))
    
    for frame_idx in range(60):  # 3 seconds at 20fps
        # Clear frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:200, :] = [200, 180, 150]
        frame[200:, :] = [80, 80, 80]
        
        # Calculate moving coordinates (moving left to right)
        shift = frame_idx * 4
        cx1, cy1, cx2, cy2 = 100 + shift, 150, 440 + shift, 380
        
        # Draw car
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (80, 120, 80), -1)
        # Windshield
        cv2.rectangle(frame, (cx1 + 30, cy1 + 10), (cx2 - 30, cy1 + 90), (220, 220, 220), -1)
        
        # Draw license plate
        px1, py1, px2, py2 = cx1 + 100, cy1 + 150, cx1 + 240, cy1 + 190
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 0), 2)
        
        # Draw plate number
        cv2.putText(
            frame, "MH12DE", (px1 + 10, py1 + 30), 
            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
        )
        
        out.write(frame)
        
    out.release()
    print(f"Saved synthetic sample video to: {VIDEO_PATH}")

def download_sample_data():
    """
    Tries to download the real-world sample video from GitHub and extract a sample image.
    Falls back to synthetic generation if offline or error occurs.
    """
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    print("Attempting to download real-world sample ANPR video from GitHub...")
    try:
        # Set a user-agent to bypass potential blocks
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        # Download the MP4 file
        urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
        print(f"Video downloaded successfully and saved to: {VIDEO_PATH}")
        
        # Extract a frame from the downloaded video to save as the sample image
        cap = cv2.VideoCapture(VIDEO_PATH)
        if cap.isOpened():
            # Skip first 100 frames to get a frame where cars are clearly visible in front of the camera
            cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(IMAGE_PATH, frame)
                print(f"Extracted sample image from video frame and saved to: {IMAGE_PATH}")
            else:
                print("Failed to read frame from video. Creating synthetic fallback image.")
                create_synthetic_data()
            cap.release()
        else:
            print("Failed to open downloaded video. Creating synthetic fallback data.")
            create_synthetic_data()
            
    except Exception as e:
        print(f"Download failed or offline: {e}")
        create_synthetic_data()

if __name__ == "__main__":
    download_sample_data()

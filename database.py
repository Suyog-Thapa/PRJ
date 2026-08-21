import sqlite3
import os
import threading
from config import DB_PATH

db_lock = threading.Lock()

def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Returns dictionaries instead of tuples
    return conn

def initialize_db():
    """
    Initializes the SQLite database tables if they do not exist.
    If the tables exist but do not match the new schema (e.g. missing vehicle_id),
    they will be dropped and re-created.
    """
    with db_lock:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if table detections exists and has the vehicle_id column
        try:
            cursor.execute("SELECT vehicle_id FROM detections LIMIT 1")
            has_vehicle_id = True
        except sqlite3.OperationalError:
            has_vehicle_id = False
            
        # Check if table detections has plate_number_devnagari
        try:
            cursor.execute("SELECT plate_number_devnagari FROM detections LIMIT 1")
            has_devnagari = True
        except sqlite3.OperationalError:
            has_devnagari = False
            
        # If table exists but schema is outdated, drop it to force recreation
        if not has_vehicle_id or not has_devnagari:
            print("Outdated database schema detected. Re-initializing tables...")
            cursor.execute("DROP TABLE IF EXISTS detections")
            cursor.execute("DROP TABLE IF EXISTS videos")
            conn.commit()
            
        # Create videos table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            location TEXT NOT NULL,
            datetime_record TEXT NOT NULL
        )
        """)
        
        # Create detections table (Refactored to support tracking and Devanagari translation)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            vehicle_id INTEGER NOT NULL,
            plate_number_english TEXT NOT NULL,
            plate_number_devnagari TEXT NOT NULL,
            timestamp TEXT NOT NULL,          -- Timeline offset format MM:SS
            confidence REAL,                  -- OCR confidence
            crop_path TEXT,                   -- Path to saved cropped plate image
            FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
            UNIQUE(video_id, vehicle_id)      -- Ensures only one record per tracked vehicle in a video
        )
        """)
        
        conn.commit()
        conn.close()
        print("Database initialized successfully.")

def add_video(filename, location, datetime_record):
    """
    Adds a video to the database or updates its details if it already exists.
    Returns the video ID. Thread-safe.
    """
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Check if video exists
            cursor.execute("SELECT id FROM videos WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            
            if row:
                video_id = row['id']
                # Update location and datetime if re-uploaded
                cursor.execute("""
                UPDATE videos 
                SET location = ?, datetime_record = ? 
                WHERE id = ?
                """, (location, datetime_record, video_id))
            else:
                cursor.execute("""
                INSERT INTO videos (filename, location, datetime_record) 
                VALUES (?, ?, ?)
                """, (filename, location, datetime_record))
                video_id = cursor.lastrowid
                
            conn.commit()
            return video_id
        finally:
            conn.close()

def add_detection_with_devnagari(video_id, vehicle_id, plate_number_english, plate_number_devnagari, timestamp, confidence, crop_path):
    """
    Upserts the best detection for a tracked vehicle in a video.
    If a record already exists, it is replaced only if the new confidence is higher.
    Thread-safe.
    """
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Check if an entry for (video_id, vehicle_id) already exists
            cursor.execute("""
            SELECT id, confidence FROM detections 
            WHERE video_id = ? AND vehicle_id = ?
            """, (video_id, vehicle_id))
            row = cursor.fetchone()
            
            if row:
                existing_id = row['id']
                existing_conf = row['confidence']
                # Update only if the new confidence is higher
                if confidence is None or existing_conf is None or confidence > existing_conf:
                    cursor.execute("""
                    UPDATE detections 
                    SET plate_number_english = ?, plate_number_devnagari = ?, 
                        timestamp = ?, confidence = ?, crop_path = ? 
                    WHERE id = ?
                    """, (plate_number_english, plate_number_devnagari, timestamp, confidence, crop_path, existing_id))
            else:
                cursor.execute("""
                INSERT INTO detections (video_id, vehicle_id, plate_number_english, plate_number_devnagari, timestamp, confidence, crop_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (video_id, vehicle_id, plate_number_english, plate_number_devnagari, timestamp, confidence, crop_path))
                
            conn.commit()
            return True
        except Exception as e:
            print(f"Error in add_detection_with_devnagari: {e}")
            return False
        finally:
            conn.close()

def search_plate(plate_number):
    """
    Searches for a plate number (space-insensitive, matches either English or Devanagari plates, supports partial match).
    Thread-safe.
    """
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Strip spaces from query term
        search_term = f"%{plate_number.replace(' ', '').upper()}%"
        
        # Use REPLACE to remove spaces from database columns during comparison
        query = """
        SELECT d.plate_number_english, d.plate_number_devnagari, d.vehicle_id, d.timestamp, d.confidence, d.crop_path,
               v.filename, v.location, v.datetime_record
        FROM detections d
        JOIN videos v ON d.video_id = v.id
        WHERE REPLACE(d.plate_number_english, ' ', '') LIKE ?
           OR REPLACE(d.plate_number_devnagari, ' ', '') LIKE ?
        ORDER BY v.datetime_record DESC, d.timestamp ASC
        """
        try:
            cursor.execute(query, (search_term, search_term))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

def get_all_records():
    """
    Retrieves all records in the database for the logs/analytics view.
    Thread-safe.
    """
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT d.id, d.vehicle_id, d.plate_number_english, d.plate_number_devnagari, d.timestamp, d.confidence, d.crop_path,
               v.filename, v.location, v.datetime_record
        FROM detections d
        JOIN videos v ON d.video_id = v.id
        ORDER BY d.id DESC
        """
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

def clear_all_records():
    """
    Deletes all records from the database and resets tables.
    Thread-safe.
    """
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM detections")
            cursor.execute("DELETE FROM videos")
            conn.commit()
            print("Database records cleared successfully.")
        finally:
            conn.close()

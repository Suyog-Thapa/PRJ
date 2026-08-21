import easyocr
import numpy as np
import re
from utils import preprocess_plate_image, clean_plate_text

# Mapping of English characters/codes to Devanagari for Nepal plates
ENGLISH_TO_DEVNAGARI = {
    # Digits
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
    # Zone / Province prefixes
    'BA': 'बा', 'PA': 'पा', 'KO': 'को', 'ME': 'मे', 'SA': 'स',
    'JA': 'ज', 'NA': 'ना', 'LU': 'लु', 'DH': 'ध', 'GA': 'ग',
    'RA': 'रा', 'BHE': 'भे', 'KA': 'क', 'SE': 'से', 'MA': 'म',
    'P1': 'प्रदेश १', 'P2': 'प्रदेश २', 'P3': 'प्रदेश ३',
    'P4': 'प्रदेश ४', 'P5': 'प्रदेश ५', 'P6': 'प्रदेश ६', 'P7': 'प्रदेश ७',
    # Vehicle categories
    'CHA': 'च', 'KHA': 'ख', 'GHA': 'घ', 'TA': 'त', 'YA': 'य',
    'BHA': 'भ', 'DA': 'द', 'HA': 'ह', 'THA': 'थ', 'A': 'अ', 'B': 'ब'
}

def correct_nepal_plate(raw_text):
    """
    Intelligent post-processing layer optimized for Nepal License Plates.
    Standard Format: [Zone/State Letters] [Lot Digits] [Category Letters] [Serial Digits]
    Example: BA 2 CHA 9831, KO 12 PA 4321
    
    It parses the characters, identifies the four constituent blocks, and resolves
    optical character confusions (e.g. O vs 0, Z vs 2, I vs 1, S vs 5) based on the block's format.
    """
    # 1. Clean space and symbols, convert to uppercase
    clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    # 2. Strip common Nepal province/state names that OCR might merge from embossed plates
    prefixes = [
        "BAGMATI", "KOSHI", "GANDAKI", "LUMBINI", "KARNALI", "SUDURPASCHIM", 
        "MADHESH", "PROVINCE", "NEPAL", "GOVERNMENT", "GOVT"
    ]
    for p in prefixes:
        clean = clean.replace(p, "")
        
    # Also strip isolated numbers at the very beginning (like "3" from "PROVINCE 3") 
    # to find the real zone letters starting point
    clean = re.sub(r'^\d+', '', clean)
    
    # Check if length matches a standard vehicle registration plate (7 to 11 chars)
    if not (7 <= len(clean) <= 11):
        return clean  # Fallback to standard clean alphanumeric if shape is odd
        
    # Mappings to resolve digit/letter ambiguities
    letter_to_digit = {
        'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 
        'B': '8', 'G': '6', 'A': '4', 'T': '7', 'Q': '0'
    }
    digit_to_letter = {
        '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', 
        '6': 'G', '4': 'A', '7': 'T', '9': 'P'
    }

    def force_letters(s):
        return "".join(digit_to_letter.get(c, c) for c in s)

    def force_digits(s):
        return "".join(letter_to_digit.get(c, c) for c in s)

    try:
        # --- BLOCK 1: Zone/State Code (First 2 characters) -> Must be letters ---
        b1 = force_letters(clean[0:2])
        
        # --- BLOCK 4: Serial Number (Last 4 characters, or 3 if plate is short) -> Must be digits ---
        serial_len = 4 if len(clean) >= 8 else 3
        b4 = force_digits(clean[-serial_len:])
        
        # --- MIDDLE PART: Lot Number + Vehicle Category Category ---
        mid = clean[2:-serial_len]
        
        b2 = ""  # Lot (Digits)
        b3 = ""  # Category (Letters)
        
        if len(mid) == 3:
            # Example: "2PA" or "2KA"
            b2 = force_digits(mid[0])
            b3 = force_letters(mid[1:])
        elif len(mid) == 4:
            # Example: "2CHA" or "12PA"
            # Look at the 2nd char in the middle part to decide if Lot is 1 or 2 digits
            c2 = mid[1]
            c2_is_digit_like = c2.isdigit() or c2 in letter_to_digit
            
            if c2_is_digit_like:
                # Lot is 2 digits, Category is 2 letters (e.g. 12PA)
                b2 = force_digits(mid[0:2])
                b3 = force_letters(mid[2:])
            else:
                # Lot is 1 digit, Category is 3 letters (e.g. 2CHA)
                b2 = force_digits(mid[0])
                b3 = force_letters(mid[1:])
        elif len(mid) == 5:
            # Example: "12CHA"
            b2 = force_digits(mid[0:2])
            b3 = force_letters(mid[2:])
        elif len(mid) == 2:
            # Example: "2PA" (with 3-digit serial)
            b2 = force_digits(mid[0])
            b3 = force_letters(mid[1])
        else:
            # Fallback for unexpected middle sizes
            b2 = force_digits(mid)
            b3 = ""

        # Re-assemble standard spacing format: "BA 2 CHA 1234"
        formatted_plate = f"{b1} {b2} {b3} {b4}".strip()
        # Verify it has standard components, otherwise return clean raw text
        if len(b1) == 2 and len(b4) >= 3 and b2:
            return formatted_plate
        return clean
    except Exception:
        return clean

def convert_to_devnagari(plate_text):
    """
    Translates a space-separated English format plate (e.g. 'BA 2 CHA 1234')
    into the corresponding Nepal Devanagari format (e.g. 'बा २ च १२३४').
    """
    if not plate_text:
        return ""
        
    parts = plate_text.split()
    if len(parts) != 4:
        # If not standard 4 parts, map character by character
        devnagari_parts = []
        for char in plate_text:
            if char in ENGLISH_TO_DEVNAGARI:
                devnagari_parts.append(ENGLISH_TO_DEVNAGARI[char])
            else:
                devnagari_parts.append(char)
        return "".join(devnagari_parts)
        
    b1, b2, b3, b4 = parts
    
    # Map Block 1 (Zone/State)
    b1_dev = ENGLISH_TO_DEVNAGARI.get(b1.upper(), b1)
    
    # Map Block 2 (Lot number digits)
    b2_dev = "".join(ENGLISH_TO_DEVNAGARI.get(d, d) for d in b2)
    
    # Map Block 3 (Category letters)
    b3_dev = ENGLISH_TO_DEVNAGARI.get(b3.upper(), b3)
    
    # Map Block 4 (Serial number digits)
    b4_dev = "".join(ENGLISH_TO_DEVNAGARI.get(d, d) for d in b4)
    
    return f"{b1_dev} {b2_dev} {b3_dev} {b4_dev}"

class LicensePlateOCR:
    def __init__(self, use_gpu=False):
        """
        Initializes the EasyOCR reader.
        """
        print("Initializing EasyOCR Reader (English)...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        print("EasyOCR initialized successfully.")

    def extract_text(self, crop_image):
        """
        Extracts plate text and runs the Nepal Intelligent Correction Layer.
        
        Returns:
            tuple: (corrected_plate_string, confidence_score)
        """
        if crop_image is None or crop_image.size == 0:
            return "", 0.0
            
        # 1. Preprocess crop using CLAHE and Bilateral Filter
        processed_img = preprocess_plate_image(crop_image)
        if processed_img is None:
            processed_img = crop_image
            
        # 2. Run OCR
        results = self.reader.readtext(processed_img, paragraph=False, detail=1)
        
        if not results:
            return "", 0.0
            
        # 3. Sort boxes from left to right
        results_sorted = sorted(results, key=lambda r: r[0][0][0])
        
        detected_texts = []
        confidences = []
        
        for bbox, text, confidence in results_sorted:
            # Clean spaces
            t_clean = text.replace(" ", "").upper()
            if t_clean:
                detected_texts.append(t_clean)
                confidences.append(confidence)
                
        if not detected_texts:
            return "", 0.0
            
        # 4. Joint text and apply Nepal Intelligence Correction Layer
        raw_combined = "".join(detected_texts)
        corrected_plate = correct_nepal_plate(raw_combined)
        
        avg_confidence = float(np.mean(confidences))
        
        return corrected_plate, avg_confidence

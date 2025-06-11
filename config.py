import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Model paths
MODEL_PATH = Path(os.getenv('MODEL_PATH', BASE_DIR / 'working' / 'runs' / 'detect' / 'yolo_car_plate' / 'weights' / 'best.pt'))  # Plate detection model
OCR_MODEL_PATH = Path(os.getenv('OCR_MODEL_PATH', BASE_DIR / 'working' / 'runs' / 'detect' / 'yolo11m_car_plate' / 'weights' / 'best.pt'))  # OCR model

# Ensure model files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Plate detection model file not found at {MODEL_PATH}")
if not OCR_MODEL_PATH.exists():
    raise FileNotFoundError(f"OCR model file not found at {OCR_MODEL_PATH}")

# Tesseract OCR path (update this based on your installation)
TESSERACT_CMD = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Detection settings
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
PLATE_CLASS_ID = int(os.getenv('PLATE_CLASS_ID', '0'))
MIN_PLATE_AREA = 1000  # Minimum area for plate detection
MAX_PLATE_AREA = 10000  # Maximum area for plate detection

# OCR settings
OCR_CONFIG = '--psm 6 --oem 3 -l ara'  # Optimized for Arabic text

# Processing settings
DRAW_DETECTIONS = True  # Whether to draw detections on the frame
SAVE_DETECTIONS = True  # Whether to save detected plates
SAVE_DIR = BASE_DIR / 'detected_plates'  # Directory to save detected plates
TEMP_DIR = Path(os.getenv('TEMP_DIR', BASE_DIR / 'temp'))  # Directory for temporary files

# Create necessary directories
SAVE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Image processing settings
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Supported image formats
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}  # Supported video formats
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '5242880'))  # 5MB
MAX_VIDEO_SIZE = int(os.getenv('MAX_VIDEO_SIZE', '104857600'))  # 100MB

# Video processing settings
VIDEO_FRAME_SKIP = int(os.getenv('VIDEO_FRAME_SKIP', '5'))  # Process every 5th frame
VIDEO_DUPLICATE_WINDOW = int(os.getenv('VIDEO_DUPLICATE_WINDOW', '30'))  # Window for duplicate detection

# OCR enhancement settings
ENHANCEMENT_SETTINGS = {
    'denoise_strength': 30,
    'denoise_template_size': 7,
    'denoise_search_size': 21,
    'sharpen_kernel': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    'adaptive_threshold_block_size': 11,
    'adaptive_threshold_c': 2
}

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://magavo1758:Jau7140LlHlghYoC@cluster0.vbg1ktw.mongodb.net/ealpr_db?retryWrites=true&w=majority')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'ealpr_db')

# Flask Configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Font Path
FONT_PATH = Path(os.getenv('FONT_PATH', BASE_DIR / 'alfont_com_arial-1.ttf')) 
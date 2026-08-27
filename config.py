import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Model paths
MODEL_PATH = Path(os.getenv('MODEL_PATH', BASE_DIR / 'working' / 'runs' / 'detect' / 'yolo_car_plate' / 'weights' / 'best.pt'))  # Plate detection model
OCR_MODEL_PATH = Path(os.getenv('OCR_MODEL_PATH', BASE_DIR / 'working' / 'runs' / 'detect' / 'yolo11m_car_plate' / 'weights' / 'best.pt'))  # OCR model

MODEL_AVAILABLE = MODEL_PATH.exists()
OCR_MODEL_AVAILABLE = OCR_MODEL_PATH.exists()
if not MODEL_AVAILABLE:
    print(f"Warning: plate detection model file not found at {MODEL_PATH}")
if not OCR_MODEL_AVAILABLE:
    print(f"Warning: OCR model file not found at {OCR_MODEL_PATH}")

# Detection settings
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
PLATE_CLASS_ID = int(os.getenv('PLATE_CLASS_ID', '0'))

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

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', '').strip()
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'ealpr_db')

# Flask Configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'ealpr-dev-secret')
DEFAULT_ADMIN_PASSWORD = os.getenv('DEFAULT_ADMIN_PASSWORD', 'admin123')
OFFLINE_ADMIN_PASSWORD = os.getenv('OFFLINE_ADMIN_PASSWORD', DEFAULT_ADMIN_PASSWORD)

# Font Path
FONT_PATH = Path(os.getenv('FONT_PATH', BASE_DIR / 'alfont_com_arial-1.ttf'))
font_path = str(FONT_PATH)

# Camera settings
CAMERA_URL = os.getenv('CAMERA_URL', '0')  # 0 for default camera
FPS_LIMIT = int(os.getenv('FPS_LIMIT', '30'))  # FPS limit for camera feed
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '640'))  # Camera frame width
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', '480'))  # Camera frame height 
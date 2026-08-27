"""
ai_models.py — YOLO model loading and shared AI constants.

Loads plate_detection_model and ocr_model once at startup.
All blueprints import from here rather than loading their own copies.
"""
from config import MODEL_PATH, OCR_MODEL_PATH

# ── Arabic character class mapping (38 classes) ──────────────────────────────
CLASS_LABELS_MAPPING = {
    0: "٠", 1: "١", 2: "٢", 3: "٣", 4: "٤", 5: "٥", 6: "٦", 7: "٧",
    8: "ح", 9: "٨", 10: "٩", 11: "ط", 12: "ظ", 13: "ع", 14: "أ", 15: "ب",
    16: "ض", 17: "د", 18: "ف", 19: "غ", 20: "ه", 21: "ج", 22: "ك", 23: "خ",
    24: "ل", 25: "م", 26: "ن", 27: "ق", 28: "ر", 29: "ص", 30: "س", 31: "ش",
    32: "ت", 33: "ث", 34: "و", 35: "ي", 36: "ذ", 37: "ز"
}

# ── Colors for bounding-box visualization ────────────────────────────────────
COLORS = [
    (255, 0, 0), (34, 75, 12), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (21, 52, 72), (66, 50, 168)
]

# ── Global camera handle (used by live video feed) ───────────────────────────
camera = None

# ── Lazy-loaded YOLO model references ────────────────────────────────────────
plate_detection_model = None
ocr_model = None
_models_loaded = False


def _load_yolo_models():
    global plate_detection_model, ocr_model, _models_loaded
    if _models_loaded:
        return
    if not MODEL_PATH.exists() or not OCR_MODEL_PATH.exists():
        print("Model files are missing; detection features will be unavailable until the models are restored.")
        _models_loaded = True
        return
    try:
        from ultralytics import YOLO

        print("Loading YOLO models...")
        plate_detection_model = YOLO(MODEL_PATH)
        ocr_model = YOLO(OCR_MODEL_PATH)
        _models_loaded = True
        print("Models loaded successfully.")
    except Exception as _e:
        print(f"Error loading models: {_e}")
        plate_detection_model = None
        ocr_model = None
        _models_loaded = True


def get_plate_detection_model():
    if plate_detection_model is None and not _models_loaded:
        _load_yolo_models()
    return plate_detection_model


def get_ocr_model():
    if ocr_model is None and not _models_loaded:
        _load_yolo_models()
    return ocr_model


def models_loaded():
    return plate_detection_model is not None and ocr_model is not None

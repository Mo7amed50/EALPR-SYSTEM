"""
ai_models.py — YOLO model loading and shared AI constants.

Loads plate_detection_model and ocr_model once at startup.
All blueprints import from here rather than loading their own copies.
"""
import threading
import time

from config import MODEL_PATH, OCR_MODEL_PATH

# Serializes YOLO inference across all request/stream threads.
# Running multiple inference calls concurrently on CPU oversubscribes the
# torch thread pool and makes the whole app unresponsive ("freezing").
INFERENCE_LOCK = threading.RLock()

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

# Phase 2 / Block C3.7: serialises *model initialisation* only.
#
# This is NOT INFERENCE_LOCK. INFERENCE_LOCK (above) guards ``model.predict`` so
# CPU inference stays one-at-a-time; this lock guards the once-per-process
# construction of the two YOLO objects. They solve different problems and must
# not be merged: holding INFERENCE_LOCK across model loading would serialise
# startup behind inference, and holding this lock during predict would defeat
# the non-freezing behaviour INFERENCE_LOCK exists to provide.
#
# Before C3 the ``if _models_loaded: return`` check was unguarded, so two
# threads calling a loader for the first time could both pass the check and both
# construct both models — briefly doubling GPU/CPU memory and discarding one
# pair. The check is now re-evaluated under this lock, so the first caller
# loads and every later caller observes ``_models_loaded`` and reuses the result.
_MODEL_LOAD_LOCK = threading.RLock()


def _load_yolo_models():
    global plate_detection_model, ocr_model, _models_loaded
    with _MODEL_LOAD_LOCK:
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
    # The outer check is a lock-free fast path only. The authoritative
    # "already loaded?" decision is re-made under _MODEL_LOAD_LOCK inside
    # _load_yolo_models, so a stale read here cannot cause a double load.
    if plate_detection_model is None and not _models_loaded:
        _load_yolo_models()
    return plate_detection_model


def get_ocr_model():
    if ocr_model is None and not _models_loaded:
        _load_yolo_models()
    return ocr_model


def predict_with_lock(model, source, **kwargs):
    """Run YOLO inference while holding a global lock (thread-safe, non-freezing)."""
    with INFERENCE_LOCK:
        if "verbose" not in kwargs:
            kwargs["verbose"] = False
        return model.predict(source, **kwargs)


def models_loaded():
    return plate_detection_model is not None and ocr_model is not None

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

# Thumbnail settings (Phase 2 / Block B).
# Thumbnails are a derived, disposable cache written to disk. They are never
# stored in MongoDB, so no schema change is involved and the original image
# data is left untouched. Deleting this directory at any time is safe: the next
# request simply regenerates what it needs.
THUMBNAIL_DIR = SAVE_DIR / 'thumbnails'
THUMBNAIL_SIZE = (320, 320)      # bounding box for gallery previews
THUMBNAIL_QUALITY = int(os.getenv('THUMBNAIL_QUALITY', '80'))

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

# ── Phase 2 / Block C5 (S1 + S3): runtime mode ────────────────────────────────
# Before C5 the entry point hard-coded `debug=True`, so the interactive debugger
# was enabled on every run regardless of intent. C5 replaces that literal with an
# explicit, validated switch and records the single-process constraint that the
# C2/C3/C4 worker design depends on.
#
# `EALPR_ENV`  development | production | test   (default: development)
# `EALPR_DEBUG` on | off                          (default: off)
#
# An unrecognised value is a configuration error, not something to guess at: this
# module raises RuntimeConfigError rather than falling back to a default, because
# silently running production with development settings is exactly the failure
# this switch exists to prevent.

VALID_ENVIRONMENTS = ('development', 'production', 'test')

_TRUTHY = {'1', 'true', 'yes', 'on'}
_FALSY = {'0', 'false', 'no', 'off', ''}


class RuntimeConfigError(RuntimeError):
    """Raised when the requested runtime mode cannot be served safely."""


def _env_flag(name, default='0'):
    """Parse a boolean environment variable, rejecting anything ambiguous."""
    raw = os.getenv(name)
    if raw is None:
        raw = default
    value = str(raw).strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise RuntimeConfigError(
        "{}={!r} is not a boolean. Use one of: {}.".format(
            name, raw, ', '.join(sorted(_TRUTHY | {'off'})))
    )


def resolve_runtime_mode():
    """Resolve ``(environment, debug)`` from the environment.

    Returns a tuple so callers cannot mutate shared state, and so the values can
    be compared directly in tests without reaching into module globals.

    Production refuses ``EALPR_DEBUG=on`` instead of quietly disabling it: a
    request that cannot be honoured must stop the process, not be downgraded into
    a different configuration than the operator asked for.
    """
    env_name = (os.getenv('EALPR_ENV') or 'development').strip().lower()
    if env_name not in VALID_ENVIRONMENTS:
        raise RuntimeConfigError(
            "EALPR_ENV={!r} is not valid. Choose one of: {}.".format(
                env_name, ', '.join(VALID_ENVIRONMENTS))
        )

    debug = _env_flag('EALPR_DEBUG')

    # The interactive debugger is available in development and nowhere else. In
    # production it is a remote code execution path; in `test` it intercepts the
    # exceptions the suite needs to see, which turns a failure into a hang. One
    # rule for both, rather than two cases that could drift apart.
    if debug and env_name != 'development':
        raise RuntimeConfigError(
            "EALPR_DEBUG=on requires EALPR_ENV=development, not "
            "EALPR_ENV={}. The interactive debugger exposes a remote code "
            "execution path in production and swallows the exceptions a test "
            "run needs to report.".format(env_name)
        )

    return env_name, debug


def assert_single_process(app_name='EALPR'):
    """Refuse a configuration that would run more than one process.

    The C2/C3/C4 inference design is process-local by construction: the queue, the
    single-consumer guarantee and the orphaned-job recovery rule all assume one
    process, and ``socketio`` runs with ``async_mode="threading"`` and no message
    queue. A second process would silently violate all three - the live Socket.IO
    broadcasts in particular would reach only the clients attached to the process
    that emitted them.

    Rather than documenting a rule that nothing enforces, the two settings that
    actually create extra processes are rejected outright. The Werkzeug reloader
    is the dangerous one: it is a single keyword argument away from being enabled,
    and it is invisible once it is.
    """
    if _env_flag('EALPR_RELOADER'):
        raise RuntimeConfigError(
            "{}: the reloader starts a second process, which this app cannot "
            "support. Its inference queue, single-consumer guarantee and live "
            "Socket.IO broadcasts are process-local. Run with one process "
            "(EALPR_RELOADER=off).".format(app_name)
        )

    workers_raw = os.getenv('EALPR_WORKERS')
    if workers_raw is not None:
        try:
            workers = int(str(workers_raw).strip())
        except ValueError:
            raise RuntimeConfigError(
                "{}: EALPR_WORKERS={!r} is not an integer.".format(app_name, workers_raw)
            )
        if workers != 1:
            raise RuntimeConfigError(
                "{}: EALPR_WORKERS={} is not supported. This app runs in a single "
                "process; see the Phase 2 C5 report for why a second process is "
                "incorrect rather than merely unsupported.".format(app_name, workers)
            ) 

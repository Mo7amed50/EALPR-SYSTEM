"""Runtime settings loaded from MongoDB with config.py fallbacks."""
from config import CONFIDENCE_THRESHOLD, CAMERA_URL, VIDEO_FRAME_SKIP, VIDEO_DUPLICATE_WINDOW


def get_confidence_threshold() -> float:
    try:
        from ealpr.extensions import DB_ENABLED
        if DB_ENABLED:
            from models import SystemSettings
            value = SystemSettings.get_setting("confidence_threshold")
            if value is not None:
                return float(value)
    except Exception:
        pass
    return CONFIDENCE_THRESHOLD


def get_camera_source():
    """Return OpenCV-compatible camera source (index int or URL string)."""
    source = str(CAMERA_URL).strip()
    if source.isdigit():
        return int(source)
    return source


def get_video_frame_skip() -> int:
    try:
        from ealpr.extensions import DB_ENABLED
        if DB_ENABLED:
            from models import SystemSettings
            value = SystemSettings.get_setting("video_frame_skip")
            if value is not None:
                return int(value)
    except Exception:
        pass
    return VIDEO_FRAME_SKIP


def get_video_duplicate_window() -> int:
    return VIDEO_DUPLICATE_WINDOW

"""Save detection images to disk instead of storing large blobs in MongoDB."""
import os
import uuid
from datetime import datetime

from config import SAVE_DIR


def save_detection_images(original_bytes: bytes, processed_bytes: bytes):
    """Persist images under detected_plates/ and return relative paths."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    original_name = f"{stamp}_{uid}_original.jpg"
    processed_name = f"{stamp}_{uid}_processed.jpg"

    original_path = SAVE_DIR / original_name
    processed_path = SAVE_DIR / processed_name

    with open(original_path, "wb") as handle:
        handle.write(original_bytes)
    with open(processed_path, "wb") as handle:
        handle.write(processed_bytes)

    return str(original_path), str(processed_path)


def read_image_bytes(path: str | None, fallback: bytes | None = None) -> bytes | None:
    if path and os.path.isfile(path):
        with open(path, "rb") as handle:
            return handle.read()
    return fallback


def encode_detection_image(path: str | None, fallback: bytes | None = None) -> str:
    data = read_image_bytes(path, fallback)
    if not data:
        return ""
    import base64
    return base64.b64encode(data).decode("utf-8")

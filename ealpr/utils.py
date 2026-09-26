"""
utils.py — Pure helper functions shared across all blueprints.

Contains image processing, Arabic text rendering, the two-stage detection
pipeline, and the live camera frame generator.
"""
import os
import time
import threading
import cv2
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from PIL import Image, ImageFont, ImageDraw
import arabic_reshaper
import bidi.algorithm

from config import FONT_PATH
from ealpr.extensions import DB_ENABLED
from ealpr.ai_models import CLASS_LABELS_MAPPING, COLORS, predict_with_lock
from ealpr.runtime_config import get_confidence_threshold, get_camera_source, get_video_frame_skip

# Guards the shared camera handle so concurrent /video_feed consumers
# never open or read the same device simultaneously (a common freeze source).
_CAMERA_LOCK = threading.Lock()

# ── Arabic font ──────────────────────────────────────────────────────────────


def _get_arabic_font(size=40):
    try:
        if FONT_PATH.exists():
            return ImageFont.truetype(str(FONT_PATH), size)
    except Exception:
        pass
    return ImageFont.load_default()


# ── Timezone conversion ──────────────────────────────────────────────────────


def utc_to_cairo(utc_time):
    """Convert a UTC datetime (or ISO string) to Africa/Cairo local time string."""
    if isinstance(utc_time, str):
        try:
            utc_time = datetime.fromisoformat(utc_time)
        except ValueError:
            return "Invalid date format"
    if utc_time is None:
        return ""
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=ZoneInfo("UTC"))
    local_time = utc_time.astimezone(ZoneInfo("Africa/Cairo"))
    return local_time.strftime('%Y-%m-%d %H:%M:%S')


# ── Arabic text rendering ────────────────────────────────────────────────────

def draw_arabic_text(image, text, position, color):
    """Draw Arabic text with a coloured background rectangle onto a NumPy image."""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = bidi.algorithm.get_display(reshaped_text)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = _get_arabic_font(40)
    bbox = draw.textbbox(position, bidi_text, font=font)
    draw.rectangle(bbox, fill=color)
    draw.text(position, bidi_text, font=font, fill="black")
    return np.array(img_pil)


def reverse_arabic(text):
    """
    Reverse Arabic text while preserving digit groups and adding spaces between
    Arabic characters and numbers.
    """
    segments = []
    current_segment = ""
    is_number = text[0].isdigit() if text else False
    for char in text:
        if char.isdigit() == is_number:
            current_segment += char
        else:
            if current_segment:
                segments.append(current_segment)
            current_segment = char
            is_number = not is_number
    if current_segment:
        segments.append(current_segment)

    reversed_text = ""
    for i, segment in enumerate(segments):
        if segment[0].isdigit():
            if i > 0:
                reversed_text += " "
            reversed_text += segment
            if i < len(segments) - 1:
                reversed_text += " "
        else:
            reversed_text = " ".join(segment[::-1]) + reversed_text
    return reversed_text.strip()


# ── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image(image):
    """Apply bilateral filter + adaptive threshold + morphological close."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return image


def enhance_plate_image(plate_image):
    """Apply CLAHE + fast NL-means denoising to a cropped plate image."""
    try:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr
    except Exception as e:
        print(f"Error in plate enhancement: {e}")
        return plate_image


# ── Core detection pipeline ──────────────────────────────────────────────────

def process_license_plate(image_path, plate_detector_model, ocr_model_ref):
    """
    Two-stage YOLO pipeline for an image FILE:
      1. Detect plate bounding box.
      2. Run OCR model on cropped plate to extract Arabic characters/digits.

    Returns:
        (plate_roi, predicted_label, success, error_msg, plate_confidence, ocr_confidence, char_details)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, False, "Failed to load image file.", 0.0, 0.0, []
    source_name = os.path.basename(image_path) or "image"
    return _detect_on_array(img, plate_detector_model, ocr_model_ref, source_name)


def _detect_on_array(img, plate_detector_model, ocr_model_ref, source_name="frame"):
    """
    Core two-stage YOLO pipeline operating directly on a BGR ndarray (no disk
    round-trip). Shared by image uploads and video/stream frame processing.
    """
    try:
        # Downscale oversized images to max 1920px width/height for fast inference
        h, w = img.shape[:2]
        max_dim = 1920
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        results = predict_with_lock(plate_detector_model, img)

        if len(results) == 0 or not results[0].boxes or len(results[0].boxes.data) == 0:
            return None, None, False, "No license plate detected.", 0.0, 0.0, []

        box = results[0].boxes.data[0].tolist()
        plate_confidence = float(box[4]) if len(box) > 4 else 0.0
        threshold = get_confidence_threshold()
        if plate_confidence < threshold:
            return None, None, False, "No license plate detected above confidence threshold.", 0.0, 0.0, []

        x1, y1, x2, y2 = [int(i) for i in box[0:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        plate_roi = img[y1:y2, x1:x2]
        if plate_roi.size == 0 or x2 <= x1 or y2 <= y1:
            return None, None, False, "Could not crop license plate area.", 0.0, 0.0, []

        plate_roi = cv2.resize(plate_roi, (220, 220))

        try:
            ocr_results = predict_with_lock(ocr_model_ref, plate_roi)
        except Exception:
            raise

        predicted_label = ""
        detected_chars_count = 0

        for result in ocr_results:
            if result.boxes and len(result.boxes.data) > 0:
                sorted_boxes = sorted(result.boxes.data, key=lambda b: b[0])
                for b in sorted_boxes:
                    x1_c, y1_c, x2_c, y2_c, conf, cls = b[:6]
                    char_conf = float(conf.item())
                    if char_conf < threshold:
                        continue
                    class_label = CLASS_LABELS_MAPPING.get(int(cls.item()), "Unknown")
                    color = COLORS[len(predicted_label) % len(COLORS)]
                    plate_roi = cv2.rectangle(
                        plate_roi,
                        (int(x1_c), int(y1_c)), (int(x2_c), int(y2_c)),
                        color, 2
                    )
                    plate_roi = draw_arabic_text(
                        plate_roi, class_label,
                        (int(x1_c), int(y1_c - 40)), color
                    )
                    predicted_label += class_label
                    detected_chars_count += 1

        if detected_chars_count == 0:
            return None, None, False, "No characters detected on the license plate.", 0.0, 0.0, []

        predicted_label = reverse_arabic(predicted_label)

        # Calculate average OCR confidence
        total_ocr_conf = 0.0
        char_details = []
        for result in ocr_results:
            if result.boxes and len(result.boxes.data) > 0:
                sorted_boxes = sorted(result.boxes.data, key=lambda b: b[0])
                for b in sorted_boxes:
                    x1_c, y1_c, x2_c, y2_c, conf, cls = b[:6]
                    char_conf = float(conf.item())
                    if char_conf < threshold:
                        continue
                    class_label = CLASS_LABELS_MAPPING.get(int(cls.item()), "Unknown")
                    char_conf = float(conf.item())
                    total_ocr_conf += char_conf
                    char_details.append({
                        "char": class_label,
                        "confidence": char_conf,
                        "box": [int(x1_c), int(y1_c), int(x2_c), int(y2_c)]
                    })
        ocr_confidence = (total_ocr_conf / detected_chars_count) if detected_chars_count > 0 else 0.0

        return plate_roi, predicted_label, True, None, plate_confidence, ocr_confidence, char_details

    except Exception as e:
        print(f"Unexpected error during license plate processing ({source_name}): {e}")
        return None, None, False, f"An unexpected error occurred: {e}", 0.0, 0.0, []


def process_frame_and_detect(frame, frame_count, plate_det_model, ocr_mdl,
                              last_detection_time=None, last_frame=None):
    """
    Process a single video frame: detect plate, run OCR, look up visitor,
    save DetectionResult, and return detection data dict.

    Returns:
        (detection_data | None, processed_img | None, current_frame)
    """
    from models import Visitor, DetectionResult
    from flask_login import current_user
    from config import VIDEO_DUPLICATE_WINDOW

    # Skip nearly-identical frames (< 1% pixel change)
    if last_frame is not None:
        frame_diff = cv2.absdiff(frame, last_frame)
        change_percent = np.sum(frame_diff > 30) / frame_diff.size * 100
        if change_percent < 1.0:
            return None, None, frame

    try:
        processed_img, plate_text, processing_success, error_msg, plate_confidence, ocr_confidence, char_details = _detect_on_array(
            frame, plate_det_model, ocr_mdl, source_name=f"frame_{frame_count}"
        )

        if processing_success:
            current_time = datetime.utcnow()

            # Duplicate suppression
            if last_detection_time and plate_text in last_detection_time:
                time_since_last = (current_time - last_detection_time[plate_text]).total_seconds()
                if time_since_last < VIDEO_DUPLICATE_WINDOW:
                    return None, None, frame

            from ealpr import extensions
            visitor = None
            if extensions.DB_ENABLED:
                try:
                    visitor = Visitor.objects(license_plate=plate_text).first()
                except Exception as db_err:
                    print(f"Error querying visitor: {db_err}")

            status = "authorized" if visitor and visitor.authorized else "unauthorized"

            visitor_info = None
            if visitor:
                visitor_info = {
                    "name": visitor.name,
                    "license_plate": visitor.license_plate,
                    "entry_time": visitor.entry_time or "N/A",
                    "exit_time": utc_to_cairo(visitor.exit_time) if visitor.exit_time else "N/A",
                    "responsible_department": visitor.responsible_department or "غير محدد",
                    "general_department": visitor.general_department or "غير محدد",
                }

            # Persist detection result if DB is online
            if extensions.DB_ENABLED:
                try:
                    from flask import has_request_context
                    from ealpr.image_storage import save_detection_images

                    _, img_encoded = cv2.imencode(".jpg", frame)
                    _, proc_encoded = cv2.imencode(".jpg", processed_img)
                    original_bytes = img_encoded.tobytes()
                    processed_bytes = proc_encoded.tobytes()
                    original_path, processed_path = save_detection_images(original_bytes, processed_bytes)

                    user_to_save = None
                    if has_request_context():
                        try:
                            from flask_login import current_user
                            if current_user and current_user.is_authenticated:
                                user_to_save = current_user
                        except Exception:
                            pass

                    detection_result = DetectionResult(
                        plate_number=plate_text,
                        confidence=plate_confidence,
                        ocr_confidence=ocr_confidence,
                        status=status,
                        timestamp=current_time,
                        visitor_name=visitor_info["name"] if visitor_info else None,
                        original_image_path=original_path,
                        processed_image_path=processed_path,
                        processed_by=user_to_save,
                    )
                    detection_result.save()
                except Exception as db_err:
                    print(f"Error saving detection result: {db_err}")

            detection_data = {
                "plate_text": plate_text,
                "status": status,
                "confidence": plate_confidence,
                "ocr_confidence": ocr_confidence,
                "char_details": char_details,
                "timestamp": utc_to_cairo(current_time),
                "visitor_info": visitor_info,
                "frame": frame_count,
            }

            if last_detection_time is not None:
                last_detection_time[plate_text] = current_time

            return detection_data, processed_img, frame

        return None, processed_img, frame

    except Exception as e:
        print(f"Error in frame processing: {e}")
        return None, None, frame


def _generate_placeholder_frame(text="Camera Offline"):
    """Generate a placeholder JPEG image with centered status text."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (26, 28, 36)
    # Draw simple border
    cv2.rectangle(img, (20, 20), (620, 460), (45, 50, 65), 2)
    # Add title
    cv2.putText(img, "EALPR SYSTEM", (210, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 160, 255), 2, cv2.LINE_AA)
    # Add status message
    cv2.putText(img, text, (170, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ok else b""


def generate_frames():
    """
    Generator that serves MJPEG frames from a shared background camera feed.
    Uses frame_id synchronization to ensure exactly 1 chunk is yielded per camera frame,
    preventing runaway CPU loops, socket flooding, and application freezing.
    """
    refs, feed = _get_or_start_feed()
    if feed is None:
        placeholder = _generate_placeholder_frame("Camera Feed Offline / Unavailable")
        try:
            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
                )
                time.sleep(1.0)
        except GeneratorExit:
            return

    last_frame_id = -1
    try:
        while not feed["stop"].is_set():
            frame_bytes = None
            with feed["condition"]:
                while feed["frame_id"] == last_frame_id and not feed["stop"].is_set():
                    if not feed["condition"].wait(timeout=0.5):
                        if feed["stop"].is_set():
                            break
                if feed["stop"].is_set():
                    break
                frame_bytes = feed["frame"]
                last_frame_id = feed["frame_id"]

            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
    except GeneratorExit:
        pass
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        _release_feed(feed)


# ── Shared camera feed plumbing ───────────────────────────────────────────────

_SHARED_FEED = {}
_FEED_STATE_LOCK = threading.Lock()


def _get_or_start_feed():
    """Return (refs, feed). Starts the background capture thread on first use."""
    global _SHARED_FEED
    from config import FPS_LIMIT, FRAME_WIDTH, FRAME_HEIGHT

    # Fast path: feed already running
    with _FEED_STATE_LOCK:
        feed = _SHARED_FEED
        if feed and feed.get("alive"):
            feed["refs"] += 1
            return feed["refs"], feed

    # ── Open the camera OUTSIDE _FEED_STATE_LOCK to avoid nested-lock deadlock ──
    camera_source = get_camera_source()
    camera = None
    with _CAMERA_LOCK:
        # On Windows, try DirectShow first for fast startup
        if isinstance(camera_source, int):
            try:
                camera = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
            except Exception:
                camera = None
        if camera is None or not camera.isOpened():
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
            camera = cv2.VideoCapture(camera_source)

        if camera is None or not camera.isOpened():
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
            print(f"Could not open camera source: {camera_source}")
            return 0, None

        # Ask the camera for a modest resolution.
        try:
            if FRAME_WIDTH > 0 and FRAME_HEIGHT > 0:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        except Exception:
            pass

    # ── Now register the feed under the state lock ───────────────────────────
    with _FEED_STATE_LOCK:
        # Another thread may have raced us; check again.
        existing = _SHARED_FEED
        if existing and existing.get("alive"):
            # We already have a live feed; release the camera we just opened.
            with _CAMERA_LOCK:
                try:
                    camera.release()
                except Exception:
                    pass
            existing["refs"] += 1
            return existing["refs"], existing

        feed = {
            "camera": camera,
            "refs": 1,
            "alive": True,
            "stop": threading.Event(),
            "condition": threading.Condition(),
            "frame": None,
            "frame_id": 0,
            "max_w": int(FRAME_WIDTH) or 640,
            "max_h": int(FRAME_HEIGHT) or 480,
            "fps_limit": min(max(int(FPS_LIMIT) or 15, 1), 30),
            "frame_skip": max(int(get_video_frame_skip()) or 1, 1),
            "last_detection_time": {},
            "frame_count": 0,
        }
        thread = threading.Thread(
            target=_feed_worker, args=(feed,), daemon=True, name="camera-feed"
        )
        feed["thread"] = thread
        _SHARED_FEED = feed
        thread.start()
        return feed["refs"], feed


def _feed_worker(feed):
    """Background capture + detection loop. Owns the camera handle."""
    import ealpr.ai_models as ai
    from ealpr.extensions import socketio

    plate_detection_model = ai.get_plate_detection_model()
    ocr_model = ai.get_ocr_model()
    camera = feed["camera"]
    min_interval = 1.0 / feed["fps_limit"]
    consecutive_failures = 0

    try:
        while not feed["stop"].is_set():
            loop_start = time.time()
            success, frame = camera.read()
            if not success or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 50:
                    print("Camera read persistently failed, stopping feed.")
                    break
                time.sleep(0.04)
                continue
            consecutive_failures = 0

            # Clamp to working resolution
            h, w = frame.shape[:2]
            if w > feed["max_w"] or h > feed["max_h"]:
                scale = min(feed["max_w"] / w, feed["max_h"] / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            feed["frame_count"] += 1
            if (
                plate_detection_model is not None and ocr_model is not None
                and feed["frame_count"] % feed["frame_skip"] == 0
            ):
                try:
                    detection_data, _processed, _f = process_frame_and_detect(
                        frame, feed["frame_count"],
                        plate_detection_model, ocr_model,
                        feed["last_detection_time"],
                    )
                    if detection_data:
                        try:
                            socketio.emit("live_detection", detection_data)
                        except Exception as emit_err:
                            print(f"Error emitting live detection data: {emit_err}")
                except Exception as det_err:
                    print(f"Detection error in feed worker: {det_err}")

            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            if ok:
                with feed["condition"]:
                    feed["frame"] = buffer.tobytes()
                    feed["frame_id"] += 1
                    feed["condition"].notify_all()

            elapsed = time.time() - loop_start
            sleep_time = min_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as worker_err:
        print(f"Camera feed worker error: {worker_err}")
    finally:
        with feed["condition"]:
            feed["frame"] = None
            feed["alive"] = False
            feed["condition"].notify_all()
        with _CAMERA_LOCK:
            try:
                if camera and camera.isOpened():
                    camera.release()
            except Exception:
                pass
        print("Camera feed stopped.")


def _release_feed(feed):
    """Decrement the consumer count and stop the feed when it reaches zero."""
    global _SHARED_FEED
    with _FEED_STATE_LOCK:
        if feed and feed.get("alive"):
            feed["refs"] = max(0, feed["refs"] - 1)
            if feed["refs"] <= 0:
                feed["stop"].set()
                feed["alive"] = False
                with feed["condition"]:
                    feed["condition"].notify_all()
                if _SHARED_FEED is feed:
                    _SHARED_FEED = {}

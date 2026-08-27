"""
utils.py — Pure helper functions shared across all blueprints.

Contains image processing, Arabic text rendering, the two-stage detection
pipeline, and the live camera frame generator.
"""
import os
import base64
import cv2
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import request
from PIL import Image, ImageFont, ImageDraw
import arabic_reshaper
import bidi.algorithm

from config import TEMP_DIR, FONT_PATH
from ealpr.extensions import DB_ENABLED
from ealpr.ai_models import CLASS_LABELS_MAPPING, COLORS
from ealpr.runtime_config import get_confidence_threshold, get_camera_source, get_video_frame_skip

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
    Two-stage YOLO pipeline:
      1. Detect plate bounding box.
      2. Run OCR model on cropped plate to extract Arabic characters/digits.

    Returns:
        (plate_roi, predicted_label, success, error_msg, plate_confidence, ocr_confidence, char_details)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None, False, "Failed to load image file.", 0.0

        results = plate_detector_model(img)
        if len(results) == 0 or not results[0].boxes or len(results[0].boxes.data) == 0:
            return None, None, False, "No license plate detected.", 0.0

        box = results[0].boxes.data[0].tolist()
        plate_confidence = float(box[4]) if len(box) > 4 else 0.0
        threshold = get_confidence_threshold()
        if plate_confidence < threshold:
            return None, None, False, "No license plate detected above confidence threshold.", 0.0

        x1, y1, x2, y2 = [int(i) for i in box[0:4]]
        plate_roi = img[y1:y2, x1:x2]
        if plate_roi.size == 0:
            return None, None, False, "Could not crop license plate area.", 0.0

        plate_roi = cv2.resize(plate_roi, (220, 220))
        temp_plate_filename = f"plate_{os.path.basename(image_path)}"
        temp_plate_path = os.path.join(TEMP_DIR, temp_plate_filename)
        cv2.imwrite(temp_plate_path, plate_roi)

        ocr_results = ocr_model_ref(temp_plate_path)
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

        if os.path.exists(temp_plate_path):
            os.remove(temp_plate_path)

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
        print(f"Unexpected error during license plate processing: {e}")
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

    temp_frame_path = os.path.join(TEMP_DIR, f"frame_{frame_count}.jpg")
    try:
        cv2.imwrite(temp_frame_path, frame)
        processed_img, plate_text, processing_success, error_msg, plate_confidence, ocr_confidence, char_details = process_license_plate(
            temp_frame_path, plate_det_model, ocr_mdl
        )

        if processing_success:
            current_time = datetime.utcnow()

            # Duplicate suppression
            if last_detection_time and plate_text in last_detection_time:
                time_since_last = (current_time - last_detection_time[plate_text]).total_seconds()
                if time_since_last < VIDEO_DUPLICATE_WINDOW:
                    return None, None, frame

            from ealpr.extensions import DB_ENABLED
            visitor = None
            if DB_ENABLED:
                try:
                    visitor = Visitor.objects(license_plate=plate_text).first()
                except Exception as db_err:
                    print(f"Error querying visitor: {db_err}")

            status = "authorized" if visitor else "unauthorized"

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
            if DB_ENABLED:
                try:
                    from ealpr.image_storage import save_detection_images

                    _, img_encoded = cv2.imencode(".jpg", frame)
                    _, proc_encoded = cv2.imencode(".jpg", processed_img)
                    original_bytes = img_encoded.tobytes()
                    processed_bytes = proc_encoded.tobytes()
                    original_path, processed_path = save_detection_images(original_bytes, processed_bytes)

                    detection_result = DetectionResult(
                        plate_number=plate_text,
                        confidence=plate_confidence,
                        ocr_confidence=ocr_confidence,
                        status=status,
                        timestamp=current_time,
                        visitor_name=visitor_info["name"] if visitor_info else None,
                        original_image_path=original_path,
                        processed_image_path=processed_path,
                        processed_by=current_user if current_user and current_user.is_authenticated else None,
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
                "processed_image": (
                    base64.b64encode(cv2.imencode(".jpg", processed_img)[1]).decode("utf-8")
                    if processed_img is not None else None
                ),
                "original_image": base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode("utf-8"),
            }

            if last_detection_time is not None:
                last_detection_time[plate_text] = current_time

            return detection_data, processed_img, frame

        return None, processed_img, frame

    except Exception as e:
        print(f"Error in frame processing: {e}")
        return None, None, frame
    finally:
        if os.path.exists(temp_frame_path):
            try:
                os.remove(temp_frame_path)
            except Exception:
                pass


def generate_frames():
    """
    Generator that reads frames from the system camera, processes every
    VIDEO_FRAME_SKIP-th frame for plate detection, and yields MJPEG bytes.
    """
    import ealpr.ai_models as ai
    from ealpr.extensions import socketio
    from ealpr.runtime_config import get_video_frame_skip

    video_frame_skip = get_video_frame_skip()
    plate_detection_model = ai.get_plate_detection_model()
    ocr_model = ai.get_ocr_model()
    if plate_detection_model is None or ocr_model is None:
        print("Models not loaded properly for camera stream.")
        return

    if ai.camera is None or not ai.camera.isOpened():
        camera_source = get_camera_source()
        ai.camera = cv2.VideoCapture(camera_source)
        if not ai.camera.isOpened():
            print(f"Could not open camera source: {camera_source}")
            return

    print("Starting camera frame generation...")
    last_detection_time = {}
    frame_count = 0

    while True:
        success, frame = ai.camera.read()
        if not success:
            print("Failed to read frame from camera stream.")
            break

        frame_count += 1
        if frame_count % video_frame_skip == 0:
            detection_data, _processed_img, _frame = process_frame_and_detect(
                frame, frame_count,
                plate_detection_model, ocr_model,
                last_detection_time
            )
            if detection_data:
                try:
                    socketio.emit("live_detection", detection_data)
                except Exception as emit_err:
                    print(f"Error emitting live detection data: {emit_err}")

        try:
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        except Exception as encode_err:
            print(f"Error encoding frame {frame_count}: {encode_err}")

    if ai.camera and ai.camera.isOpened():
        ai.camera.release()
        print("Camera released.")

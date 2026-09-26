"""
blueprints/detection.py — License plate detection, video processing, and history.

Routes:
    POST /process_plate
    POST /process_video
    GET  /video_feed
    GET  /detection_history
    GET  /uploaded_images
    GET  /api/detection/<detection_id>
    POST /api/detections/clear
    GET  /api/export/detections
    GET  /api/images/<detection_id>
    GET  /api/activities
"""
import io
import os
import csv
import base64
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, jsonify, Response, flash, send_file
from flask_login import login_required, current_user
from mongoengine import Q
from bson.objectid import ObjectId, InvalidId
from werkzeug.utils import secure_filename

from ealpr import extensions
from ealpr.decorators import db_required
from ealpr.image_storage import save_detection_images, encode_detection_image, get_image_file_path, read_image_bytes
from config import (
    ALLOWED_EXTENSIONS, TEMP_DIR,
    ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_SIZE, MAX_IMAGE_SIZE,
)
from models import Visitor, DetectionResult
from ealpr.ai_models import get_plate_detection_model, get_ocr_model
from ealpr.extensions import socketio
from ealpr.utils import (
    utc_to_cairo,
    process_license_plate,
    process_frame_and_detect,
    generate_frames,
)
from ealpr.runtime_config import get_video_frame_skip

detection_bp = Blueprint("detection", __name__)


# Phase 2 / Block A — detection images are private, authenticated content.
#
# These were previously served as "public, max-age=3600", which allows shared and
# intermediary caches (corporate proxies, some CDNs) to store and replay plate
# images for other users. They are now marked "private" so only the browser's own
# cache may retain them, with Vary on Cookie so a cache cannot serve one
# authenticated user's image to another. max-age is kept short because a detection
# record's image content is immutable once written.
_AUTHENTICATED_IMAGE_CACHE_CONTROL = "private, max-age=300, must-revalidate"
_IMAGE_CACHE_VARY = "Cookie"


def _apply_private_image_cache(response):
    """Attach private cache headers to an authenticated image response."""
    response.headers["Cache-Control"] = _AUTHENTICATED_IMAGE_CACHE_CONTROL
    response.headers["Vary"] = _IMAGE_CACHE_VARY
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response



# ── Image upload & plate detection ───────────────────────────────────────────

@detection_bp.route("/process_plate", methods=["POST"])
@login_required
def process_plate():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    image = request.files["image"]
    safe_filename = secure_filename(image.filename or "")
    if not safe_filename:
        return jsonify({"success": False, "error": "No selected file"}), 400
    if not safe_filename.lower().endswith(tuple(f".{ext}" for ext in ALLOWED_EXTENSIONS)):
        return jsonify({
            "success": False,
            "error": f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    image.stream.seek(0, os.SEEK_END)
    image_size = image.stream.tell()
    image.stream.seek(0)
    if image_size > MAX_IMAGE_SIZE:
        return jsonify({
            "success": False,
            "error": f"Image file size exceeds limit of {MAX_IMAGE_SIZE / 1024 / 1024:.2f} MB"
        }), 400

    plate_detection_model = get_plate_detection_model()
    ocr_model = get_ocr_model()
    if plate_detection_model is None or ocr_model is None:
        return jsonify({"success": False, "error": "Models not loaded properly"}), 503

    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4().hex}_{safe_filename}")

    try:
        image.save(temp_path)
        processed_img, plate_text, success, error_msg, plate_confidence, ocr_confidence, char_details = process_license_plate(
            temp_path, plate_detection_model, ocr_model
        )

        if not success:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"success": False, "message": error_msg, "status": "error"})

        import cv2
        encoded, buffer = cv2.imencode(".jpg", processed_img)
        if not encoded:
            return jsonify({"success": False, "message": "Could not encode processed image"}), 500
        processed_img_base64 = base64.b64encode(buffer).decode("utf-8")

        visitor = None
        status = "unknown"
        visitor_name = None
        visitor_info = None

        if extensions.DB_ENABLED:
            try:
                visitor = Visitor.objects(license_plate=plate_text).first()
                status = "authorized" if visitor and visitor.authorized else "unauthorized"
                visitor_name = visitor.name if visitor else None

                # Auto-set entry time on first detection
                if visitor and not visitor.entry_datetime_utc:
                    try:
                        visitor.entry_datetime_utc = datetime.utcnow()
                        visitor.save()
                    except Exception as upd_err:
                        print(f"Error setting entry time for {visitor.name}: {upd_err}")

                if visitor:
                    entry_time_str = (
                        visitor.entry_time if isinstance(visitor.entry_time, str)
                        else utc_to_cairo(visitor.entry_time) if isinstance(visitor.entry_time, datetime)
                        else None
                    )
                    exit_time_str = (
                        visitor.exit_time if isinstance(visitor.exit_time, str)
                        else utc_to_cairo(visitor.exit_time) if isinstance(visitor.exit_time, datetime)
                        else None
                    )
                    visitor_info = {
                        "name": visitor.name,
                        "visitor_id": str(visitor.visitor_id),
                        "license_plate": visitor.license_plate,
                        "entry_time": entry_time_str,
                        "exit_time": exit_time_str,
                        "authorized": visitor.authorized,
                        "status": visitor.status,
                        "responsible_department": visitor.responsible_department,
                        "general_department": visitor.general_department,
                    }
            except Exception as db_err:
                print(f"Error querying visitor: {db_err}")

        detection_timestamp = datetime.utcnow()
        if extensions.DB_ENABLED:
            try:
                with open(temp_path, "rb") as original_file:
                    original_bytes = original_file.read()
                processed_bytes = buffer.tobytes()
                original_path, processed_path = save_detection_images(original_bytes, processed_bytes)

                detection_record = DetectionResult(
                    plate_number=plate_text,
                    confidence=plate_confidence,
                    ocr_confidence=ocr_confidence,
                    status=status,
                    visitor_name=visitor_name,
                    processed_by=current_user if current_user.is_authenticated else None,
                    original_image_path=original_path,
                    processed_image_path=processed_path,
                    timestamp=detection_timestamp,
                )
                detection_record.save()
            except Exception as db_err:
                print(f"Error saving detection record: {db_err}")

        display_confidence = float(plate_confidence) if isinstance(plate_confidence, (int, float)) else 0.0

        return jsonify({
            "success": True,
            "plate_text": plate_text,
            "status": status,
            "visitor_name": visitor_name,
            "processed_image": processed_img_base64,
            "confidence": display_confidence,
            "ocr_confidence": ocr_confidence,
            "char_details": char_details,
            "visitor_id": str(visitor.id) if visitor else None,
            "visitor_info": visitor_info,
            "detection_timestamp": utc_to_cairo(detection_timestamp) if detection_timestamp else None,
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"success": False, "error": "Image processing failed"}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as cleanup_err:
            print(f"Error cleaning up temp file: {cleanup_err}")


# ── Video upload & batch processing (async) ─────────────────────────────────
#
# Video processing runs in a background thread so it never blocks the Flask
# request thread.  The client receives a job_id immediately and polls
# /api/video_job/<job_id> for progress and results.

import threading as _vthread

_video_jobs = {}
_video_jobs_lock = _vthread.Lock()


def _run_video_job(job_id, temp_path, plate_detection_model, ocr_model, app):
    """Background thread: process video frames and update job state."""
    import cv2 as _cv2
    with app.app_context():
        with _video_jobs_lock:
            _video_jobs[job_id]["status"] = "running"
        cap = None
        try:
            cap = _cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")

            total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_count = 0
            processed_frames = 0
            best_detections = {}
            last_frame = None
            last_detection_time = {}

            base_skip = max(int(get_video_frame_skip()), 1)
            frame_skip = max(base_skip, total_frames // 200) if total_frames > 1000 else base_skip

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_skip == 0:
                    detection_data, _proc, current_frame = process_frame_and_detect(
                        frame, frame_count, plate_detection_model, ocr_model,
                        last_detection_time, last_frame
                    )
                    if detection_data:
                        plate_text = detection_data["plate_text"]
                        if (plate_text not in best_detections or
                                detection_data["confidence"] > best_detections[plate_text]["confidence"]):
                            best_detections[plate_text] = detection_data
                            try:
                                socketio.emit("live_detection", detection_data)
                            except Exception:
                                pass
                    processed_frames += 1
                    last_frame = current_frame
                    if total_frames > 0:
                        progress = min(int(frame_count / total_frames * 100), 99)
                        with _video_jobs_lock:
                            _video_jobs[job_id]["progress"] = progress

            cap.release()
            cap = None
            with _video_jobs_lock:
                _video_jobs[job_id].update({
                    "status": "done",
                    "progress": 100,
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "detections": list(best_detections.values()),
                })
        except Exception as e:
            with _video_jobs_lock:
                _video_jobs[job_id].update({"status": "error", "error": str(e)})
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass


@detection_bp.route("/process_video", methods=["POST"])
@login_required
def process_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    video = request.files["video"]
    safe_filename = secure_filename(video.filename or "")
    if not safe_filename:
        return jsonify({"success": False, "error": "No selected file"}), 400

    file_ext = safe_filename.rsplit(".", 1)[1].lower() if "." in safe_filename else ""
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f'Unsupported file type. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}'
        }), 400

    video.seek(0, os.SEEK_END)
    file_size = video.tell()
    video.seek(0)
    if file_size > MAX_VIDEO_SIZE:
        return jsonify({
            "success": False,
            "error": f"Video file size exceeds limit of {MAX_VIDEO_SIZE / 1024 / 1024:.2f} MB"
        }), 400

    plate_detection_model = get_plate_detection_model()
    ocr_model = get_ocr_model()
    if plate_detection_model is None or ocr_model is None:
        return jsonify({"success": False, "error": "Models not loaded properly"}), 503

    from flask import current_app
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4().hex}_{safe_filename}")
    try:
        video.save(temp_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save upload: {e}"}), 500

    job_id = uuid.uuid4().hex
    with _video_jobs_lock:
        _video_jobs[job_id] = {"status": "pending", "progress": 0}

    app_obj = current_app._get_current_object()
    t = _vthread.Thread(
        target=_run_video_job,
        args=(job_id, temp_path, plate_detection_model, ocr_model, app_obj),
        daemon=True,
        name=f"video-job-{job_id[:8]}",
    )
    t.start()
    return jsonify({"success": True, "job_id": job_id, "message": "Video processing started"})


@detection_bp.route("/api/video_job/<string:job_id>")
@login_required
def get_video_job(job_id):
    """Poll endpoint: returns current status/progress for an async video job."""
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, **job})



# ── Live camera MJPEG stream ─────────────────────────────────────────────────

@detection_bp.route("/video_feed")
@login_required
def video_feed():
    plate_detection_model = get_plate_detection_model()
    ocr_model = get_ocr_model()
    if plate_detection_model is None or ocr_model is None:
        return Response("Models not loaded properly", status=500)
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── Detection history (paginated) ────────────────────────────────────────────

@detection_bp.route("/detection_history")
@login_required
@db_required
def detection_history():
    page = max(request.args.get("page", 1, type=int) or 1, 1)
    per_page = 10
    skip = (page - 1) * per_page
    total = DetectionResult.objects.count()
    detections = DetectionResult.objects.order_by("-timestamp")[skip: skip + per_page]
    total_pages = (total + per_page - 1) // per_page
    return render_template(
        "detection_history.html",
        detections=detections,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        max=max,
        min=min,
    )


# ── Uploaded images gallery ──────────────────────────────────────────────────

# Phase 2 / Block A. The gallery previously rendered every DetectionResult with
# DetectionResult.objects.order_by("-timestamp").all(), which fetched the whole
# collection on each page load. Only the requested page is now queried.
#
# The collection still mixes two image storage layouts (legacy BinaryField and
# newer *_image_path). Documents are therefore read with .only() so MongoDB never
# transfers the binary image payloads just to render a list of cards; the images
# themselves are still fetched one at a time by /api/images/<id>/<type>.
GALLERY_ALLOWED_PAGE_SIZES = (20, 50, 100)
GALLERY_DEFAULT_PAGE_SIZE = 20
# Fields needed to render a card. Image payloads are deliberately excluded.
GALLERY_LIST_FIELDS = (
    "timestamp", "plate_number", "confidence", "status",
    "visitor_name", "original_image_path", "processed_image_path",
)


def _parse_positive_int(raw, default, allowed=None):
    """Parse a positive int from a query arg, falling back safely."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value < 1:
        return default
    if allowed is not None and value not in allowed:
        # Snap to the nearest allowed size rather than trusting arbitrary input.
        return min(allowed, key=lambda size: abs(size - value))
    return value


@detection_bp.route("/uploaded_images")
@login_required
@db_required
def uploaded_images():
    try:
        requested_page = request.args.get("page")
        requested_size = request.args.get("per_page")

        per_page = _parse_positive_int(
            requested_size, GALLERY_DEFAULT_PAGE_SIZE, GALLERY_ALLOWED_PAGE_SIZES
        )

        query = Q()

        # Preserve the existing search / status / date filter controls, but apply
        # them in the database so they still work across every page.
        search_query = (request.args.get("q") or "").strip()
        if search_query:
            query &= Q(plate_number__icontains=search_query)

        status_filter = (request.args.get("status") or "all").strip()
        if status_filter in ("authorized", "unauthorized"):
            query &= Q(status=status_filter)

        date_filter = (request.args.get("date_filter") or "all").strip()
        now = datetime.utcnow()
        if date_filter == "today":
            query &= Q(timestamp__gte=now.replace(hour=0, minute=0, second=0, microsecond=0))
        elif date_filter == "week":
            query &= Q(timestamp__gte=now - timedelta(days=7))
        elif date_filter == "month":
            query &= Q(timestamp__gte=now - timedelta(days=30))

        total = DetectionResult.objects(query).count()

        total_pages = max(1, (total + per_page - 1) // per_page)
        page = _parse_positive_int(requested_page, 1)
        if page > total_pages:
            page = total_pages

        # -timestamp uses the existing timestamp_1 index; -id is a stable
        # tiebreaker so records sharing a timestamp cannot shift between pages.
        detections = (
            DetectionResult.objects(query)
            .only(*GALLERY_LIST_FIELDS)
            .order_by("-timestamp", "-id")
            .skip((page - 1) * per_page)
            .limit(per_page)
        )

        return render_template(
            "uploaded_images.html",
            detections=detections,
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
            search_query=search_query,
            status_filter=status_filter,
            date_filter=date_filter,
            allowed_page_sizes=GALLERY_ALLOWED_PAGE_SIZES,
        )
    except Exception as e:
        print(f"Error retrieving detections: {e}")
        flash("An error occurred while loading images. Please try again.", "error")
        return render_template(
            "uploaded_images.html",
            detections=[],
            page=1,
            per_page=GALLERY_DEFAULT_PAGE_SIZE,
            total=0,
            total_pages=1,
            search_query="",
            status_filter="all",
            date_filter="all",
            allowed_page_sizes=GALLERY_ALLOWED_PAGE_SIZES,
        )


# ── Single detection detail (JSON) ──────────────────────────────────────────

@detection_bp.route("/api/detection/<string:detection_id>")
@login_required
@db_required
def get_detection(detection_id):
    from bson.objectid import InvalidId
    if not ObjectId.is_valid(detection_id):
        return jsonify({"success": False, "message": "Invalid detection ID format"}), 400
    try:
        detection = DetectionResult.objects(id=ObjectId(detection_id)).first()
        if not detection:
            return jsonify({"success": False, "message": "Detection record not found"}), 404

        processed_by_username = detection.processed_by.username if detection.processed_by else None
        visitor = Visitor.objects(license_plate=detection.plate_number).first() if detection.plate_number else None

        if visitor:
            entry_time_str = (
                visitor.entry_time if isinstance(visitor.entry_time, str)
                else utc_to_cairo(visitor.entry_time) if isinstance(visitor.entry_time, datetime)
                else None
            )
            exit_time_str = (
                visitor.exit_time if isinstance(visitor.exit_time, str)
                else utc_to_cairo(visitor.exit_time) if isinstance(visitor.exit_time, datetime)
                else None
            )
            visitor_info = {
                "name": visitor.name or "N/A",
                "visitor_id": str(visitor.visitor_id) if visitor.visitor_id else "N/A",
                "license_plate": visitor.license_plate,
                "entry_time": entry_time_str or "N/A",
                "exit_time": exit_time_str or "N/A",
                "department": visitor.responsible_department or "N/A",
            }
        else:
            visitor_info = {
                "name": detection.visitor_name or "N/A",
                "visitor_id": "N/A",
                "license_plate": detection.plate_number,
                "entry_time": "N/A",
                "exit_time": "N/A",
                "department": "N/A",
            }

        timestamp_str = (
            detection.timestamp if isinstance(detection.timestamp, str)
            else utc_to_cairo(detection.timestamp) if isinstance(detection.timestamp, datetime)
            else None
        )

        return jsonify({
            "success": True,
            "id": str(detection.id),
            "timestamp": timestamp_str,
            "plate_number": detection.plate_number,
            "confidence": float(detection.confidence) if detection.confidence is not None else 0.0,
            "status": detection.status,
            "visitor_name": detection.visitor_name,
            "processed_by": processed_by_username,
            "original_image_url": f"/api/images/{detection_id}/original",
            "processed_image_url": f"/api/images/{detection_id}/processed",
            "original_image": encode_detection_image(detection.original_image_path, detection.original_image),
            "processed_image": encode_detection_image(detection.processed_image_path, detection.processed_image),
            "visitor_info": visitor_info,
        })

    except Exception as e:
        print(f"Error fetching detection {detection_id}: {e}")
        return jsonify({"success": False, "message": f"Error fetching detection details: {e}"}), 500


# ── Clear all detection history (admin) ─────────────────────────────────────

@detection_bp.route("/api/detections/clear", methods=["POST"])
@login_required
@db_required
def clear_detection_history():
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        DetectionResult.objects.delete()
        return jsonify({"success": True, "message": "Detection history cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to clear detection history: {e}"}), 500


# ── Export detections as CSV or Excel ────────────────────────────────────────

@detection_bp.route("/api/export/detections")
@login_required
@db_required
def export_detections():
    try:
        status_filter = request.args.get("status", "all")
        date_filter = request.args.get("date_filter", "all")
        fmt = request.args.get("format", "csv")

        query = DetectionResult.objects
        if status_filter != "all":
            query = query.filter(status=status_filter)
        if date_filter != "all":
            now = datetime.utcnow()
            if date_filter == "today":
                query = query.filter(timestamp__gte=now.replace(hour=0, minute=0, second=0))
            elif date_filter == "week":
                query = query.filter(timestamp__gte=now - timedelta(days=7))
            elif date_filter == "month":
                query = query.filter(timestamp__gte=now - timedelta(days=30))

        results = query.order_by("-timestamp")
        if results.count() == 0:
            return jsonify({"error": "No data found for the selected filters"}), 404

        if fmt == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Timestamp", "Plate Number", "Status", "Confidence", "Visitor Name", "Processed By"])
            for r in results:
                processed_by_username = r.processed_by.username if r.processed_by else "N/A"
                writer.writerow([
                    utc_to_cairo(r.timestamp),
                    r.plate_number,
                    r.status,
                    f"{r.confidence * 100:.2f}%" if r.confidence is not None else "N/A",
                    r.visitor_name or "N/A",
                    processed_by_username,
                ])
            output.seek(0)
            return Response(
                output,
                mimetype="text/csv",
                headers={"Content-Disposition": f'attachment; filename=detections_export_{datetime.utcnow().strftime("%Y%m%d")}.csv'},
            )

        elif fmt == "excel":
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "Detection History"
            headers = ["Timestamp", "Plate Number", "Status", "Confidence", "Visitor Name", "Processed By"]
            for col, header in enumerate(headers, 1):
                ws.cell(row=1, column=col, value=header)
            for row_idx, r in enumerate(results, 2):
                processed_by_username = r.processed_by.username if r.processed_by else "N/A"
                ws.cell(row=row_idx, column=1, value=utc_to_cairo(r.timestamp))
                ws.cell(row=row_idx, column=2, value=r.plate_number)
                ws.cell(row=row_idx, column=3, value=r.status)
                ws.cell(row=row_idx, column=4, value=f"{r.confidence * 100:.2f}%" if r.confidence is not None else "N/A")
                ws.cell(row=row_idx, column=5, value=r.visitor_name or "N/A")
                ws.cell(row=row_idx, column=6, value=processed_by_username)
            excel_file = io.BytesIO()
            wb.save(excel_file)
            excel_file.seek(0)
            return Response(
                excel_file,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename=detections_export_{datetime.utcnow().strftime("%Y%m%d")}.xlsx'},
            )
        else:
            return jsonify({"error": "Unsupported export format"}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to export data: {e}"}), 500


# ── Single detection image retrieval ─────────────────────────────────────────

@detection_bp.route("/api/images/<string:detection_id>/<string:img_type>")
@login_required
@db_required
def serve_detection_image_file(detection_id, img_type):
    if not ObjectId.is_valid(detection_id):
        return jsonify({"success": False, "message": "Invalid detection ID format."}), 400
    try:
        detection = DetectionResult.objects(id=ObjectId(detection_id)).first()
        if not detection:
            return jsonify({"success": False, "message": "Detection record not found."}), 404

        target_path = detection.original_image_path if img_type == "original" else detection.processed_image_path
        target_bytes = detection.original_image if img_type == "original" else detection.processed_image

        file_path = get_image_file_path(target_path)
        if file_path:
            res = send_file(file_path, mimetype="image/jpeg")
            return _apply_private_image_cache(res)

        raw_bytes = target_bytes
        if raw_bytes:
            res = Response(raw_bytes, mimetype="image/jpeg")
            return _apply_private_image_cache(res)

        return jsonify({"success": False, "message": "Image content not available"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@detection_bp.route("/api/images/<string:detection_id>")
@login_required
@db_required
def get_detection_image(detection_id):
    if not ObjectId.is_valid(detection_id):
        return jsonify({"success": False, "message": "Invalid detection ID format."}), 400
    try:
        detection = DetectionResult.objects(id=ObjectId(detection_id)).first()
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid detection ID format."}), 400

    if not detection:
        return jsonify({"success": False, "message": "Detection record not found."}), 404

    return jsonify({
        "success": True,
        "original_image_url": f"/api/images/{detection_id}/original",
        "processed_image_url": f"/api/images/{detection_id}/processed",
        "original_image": "",
        "processed_image": "",
        "plate_number": detection.plate_number,
        "status": detection.status,
        "timestamp": utc_to_cairo(detection.timestamp) if detection.timestamp else None,
        "visitor_name": detection.visitor_name,
    })



# ── Recent activity feed (last 10 detections) ─────────────────────────────

@detection_bp.route("/api/activities")
@login_required
@db_required
def get_activities():
    try:
        activities = DetectionResult.objects.order_by("-timestamp").limit(10)
        activity_list = [
            {
                "action": "detect_plate",
                "details": f"Detected plate {a.plate_number} - {a.status}",
                "timestamp": utc_to_cairo(a.timestamp),
            }
            for a in activities
        ]
        return jsonify({"success": True, "activities": activity_list})
    except Exception as e:
        return jsonify({"success": False, "error": "Failed to fetch activities"}), 500


# ── Plate number search API ───────────────────────────────────────────────

@detection_bp.route("/api/search/plate")
@login_required
@db_required
def search_plate():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"success": False, "message": "Search query required"}), 400
    try:
        # Search visitor database
        visitor = Visitor.objects(Q(license_plate__icontains=query) | Q(name__icontains=query)).first()
        if visitor:
            return jsonify({
                "success": True,
                "data": {
                    "id": str(visitor.id),
                    "name": visitor.name,
                    "license_plate": visitor.license_plate,
                    "status": "authorized" if visitor.authorized else "unauthorized"
                }
            })
        return jsonify({"success": False, "message": "No visitor found"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ── SocketIO connection events ───────────────────────────────────────────────


@socketio.on("connect")
def handle_connect():
    print("Client connected to SocketIO.")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected from SocketIO.")

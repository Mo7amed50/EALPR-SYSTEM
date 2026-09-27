"""
blueprints/inference.py — Phase 2 / Block C2: asynchronous inference job API.

Two endpoints, both authenticated:

    POST /api/inference/jobs          enqueue an image for background inference
    GET  /api/inference/jobs/<job_id> read job status / result

The POST endpoint performs NO inference. It validates and stages the upload,
creates a queued job and returns HTTP 202 immediately. The expensive YOLO/OCR
work happens on the background worker from C2.3.

``/process_plate`` is untouched and remains the synchronous compatibility path.
"""

import os
import uuid

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE, TEMP_DIR
from ealpr import inference_jobs
from ealpr.decorators import db_required
from models import InferenceJob

inference_bp = Blueprint("inference", __name__)

# Public, non-sensitive error text for the submission endpoint.
_SUBMIT_ERRORS = {
    "no_file": "No image provided",
    "no_filename": "No selected file",
    "bad_type": "Unsupported file type. Allowed: {}".format(
        ", ".join(sorted(ALLOWED_EXTENSIONS))),
    "too_large": "Image file size exceeds limit of {:.2f} MB".format(
        MAX_IMAGE_SIZE / 1024 / 1024),
    "stage_failed": "Could not stage the uploaded image.",
    "models_unavailable": "Models not loaded properly",
}


def _error(code, status):
    return jsonify({"success": False, "error": _SUBMIT_ERRORS[code],
                    "error_code": code}), status


@inference_bp.route("/api/inference/jobs", methods=["POST"])
@login_required
@db_required
def create_inference_job():
    """Accept an image and queue it. Returns 202 without running inference."""
    if "image" not in request.files:
        return _error("no_file", 400)

    image = request.files["image"]
    safe_filename = secure_filename(image.filename or "")
    if not safe_filename:
        return _error("no_filename", 400)
    if not safe_filename.lower().endswith(
            tuple(".{}".format(ext) for ext in ALLOWED_EXTENSIONS)):
        return _error("bad_type", 400)

    image.stream.seek(0, os.SEEK_END)
    image_size = image.stream.tell()
    image.stream.seek(0)
    if image_size > MAX_IMAGE_SIZE:
        return _error("too_large", 400)

    # Same availability gate as /process_plate. Loading happens here, in the
    # request thread, so a misconfigured deployment fails fast with 503 rather
    # than accepting jobs that can only ever fail.
    from ealpr.ai_models import get_ocr_model, get_plate_detection_model
    if get_plate_detection_model() is None or get_ocr_model() is None:
        return _error("models_unavailable", 503)

    # Stage the upload for the worker, which reads it from disk exactly as
    # process_plate does. The path is server-side state and is never returned.
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(
        TEMP_DIR, "inference_job_{}_{}".format(uuid.uuid4().hex, safe_filename)
    )
    try:
        image.save(temp_path)
    except Exception:
        return _error("stage_failed", 400)

    try:
        job = inference_jobs.enqueue(
            current_app._get_current_object(),
            temp_path,
            safe_filename,
            current_user._get_current_object(),
        )
    except Exception:
        # The job never entered the queue, so nothing will clean the file up.
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return _error("stage_failed", 503)

    return jsonify({
        "success": True,
        "job_id": job.job_id,
        "status": job.status,
    }), 202


@inference_bp.route("/api/inference/jobs/<string:job_id>")
@login_required
@db_required
def get_inference_job(job_id):
    """Return job status. Only the owning user (or an admin) may read it."""
    job = InferenceJob.objects(job_id=job_id).first()
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404

    # Ownership. A non-owner gets 404 rather than 403 so the endpoint cannot be
    # used to probe which job ids exist. This matches the "not found" convention
    # used elsewhere in the app.
    owner_id = job.user.id if job.user else None
    is_owner = owner_id is not None and owner_id == current_user.id
    if not (is_owner or getattr(current_user, "is_admin", False)):
        return jsonify({"success": False, "error": "Job not found"}), 404

    payload = {
        "success": True,
        "job_id": job.job_id,
        "status": job.status,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "completed_at": _iso(job.completed_at),
        "detection_id": str(job.detection.id) if job.detection else None,
    }

    if job.status == InferenceJob.STATUS_COMPLETED:
        payload["plate_text"] = job.plate_text
    if job.status == InferenceJob.STATUS_FAILED:
        # Safe category text only. No traceback, no path, no driver message.
        payload["error"] = job.error_message
        payload["error_code"] = job.error_code

    return jsonify(payload)


def _iso(value):
    """Render a timestamp for the API. Naive datetimes are UTC in this codebase."""
    if value is None:
        return None
    try:
        return value.replace(microsecond=0).isoformat() + "Z"
    except AttributeError:
        return str(value)

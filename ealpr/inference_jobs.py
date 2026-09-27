"""
inference_jobs.py — Phase 2 / Block C2: background inference worker.

WHY AN IN-PROCESS THREAD AND NOT CELERY
----------------------------------------
Block C2.1 audit found no queue infrastructure installed (no Redis, Celery, RQ,
Flask-Executor). A broker was considered and rejected on the evidence:

1. ``ai_models.INFERENCE_LOCK`` is an RLock that serialises every YOLO call.
   A distributed worker fleet would therefore queue on the *same* lock and
   still execute one inference at a time. Celery buys concurrency the lock
   forbids, so it would add a broker, a new failure mode and operational
   surface for no throughput gain.
2. The app runs as a single process (Werkzeug dev server via ``socketio.run``,
   ``use_reloader=False``). Dispatching work to *processes* would load a second
   YOLO instance per worker — directly contradicting the C2.9 requirement not
   to create one model instance per job, and wasteful of GPU memory.
3. The project already has this exact pattern: ``_run_video_job`` runs in a
   ``threading.Thread`` inside ``app.app_context()``. Reusing it introduces no
   new operational concept.

So: ONE consumer thread, a FIFO ``queue.Queue``, and durable state in MongoDB.
Job state lives in the ``inference_job`` collection rather than a process-local
dict, so status survives a restart and ownership is enforceable — two
weaknesses of the existing in-memory ``_video_jobs`` store that this block does
not inherit.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not reimplement the AI pipeline. Inference is delegated verbatim to
``ealpr.utils.process_license_plate`` — the same function ``/process_plate``
calls, which acquires ``INFERENCE_LOCK`` internally via ``predict_with_lock``.
Detection persistence mirrors ``process_plate`` field for field. No weights,
thresholds, OCR preprocessing, Arabic handling or confidence maths are touched.
"""

import logging
import os
import queue
import threading
import traceback
import uuid
from datetime import datetime

from ealpr import extensions
from ealpr.ai_models import get_ocr_model, get_plate_detection_model
from ealpr.image_storage import save_detection_images
from ealpr.utils import process_license_plate, utc_to_cairo
from models import DetectionResult, InferenceJob, Visitor

log = logging.getLogger(__name__)


# Failure categories. These are stable, machine-readable and safe to expose.
ERROR_NO_PLATE = "no_plate_detected"
ERROR_INFERENCE = "inference_failed"
ERROR_STORAGE = "image_storage_failed"
ERROR_PERSIST = "detection_persist_failed"
ERROR_INTERNAL = "internal_error"

# Safe, non-leaking client messages.
_SAFE_MESSAGES = {
    ERROR_NO_PLATE: "No license plate was detected in the uploaded image.",
    ERROR_INFERENCE: "The image could not be processed for plate recognition.",
    ERROR_STORAGE: "The uploaded image could not be read for storage.",
    ERROR_PERSIST: "The detection result could not be saved.",
    ERROR_INTERNAL: "The inference job failed unexpectedly.",
}

_queue = queue.Queue()
_queue_lock = threading.Lock()
_worker_thread = None
_shutdown = threading.Event()
# Idle notification used by the test-suite to await deterministic drain.
_idle = threading.Event()
_idle.set()


# ── Lifecycle ────────────────────────────────────────────────────────────────

def _ensure_worker(app):
    """Start the single consumer thread on first use."""
    global _worker_thread
    with _queue_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        _shutdown.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(app,),
            daemon=True,
            name="inference-worker",
        )
        _worker_thread.start()
        return _worker_thread


def _worker_loop(app):
    """Consume jobs one at a time, forever."""
    while not _shutdown.is_set():
        try:
            job_id = _queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if job_id is None:  # shutdown sentinel
            _queue.task_done()
            break
        try:
            run_job(app, job_id)
        except Exception:
            # Belt and braces: run_job owns its own error handling, so reaching
            # here means something failed outside it. The worker must survive.
            log.exception("Inference worker loop caught an unexpected error "
                          "for job %s", job_id)
        finally:
            _queue.task_done()
            # Only advertise idleness once the backlog is genuinely gone, so a
            # waiter cannot observe a transient empty moment mid-batch.
            if _queue.empty():
                _idle.set()


def enqueue(app, image_path, original_filename, user, start_worker=True):
    """Create a queued job and hand it to the worker. Returns the job document.

    Does no inference. The client gets control back immediately.

    ``start_worker=False`` stages the job without spinning up the consumer
    thread, which is how the test-suite observes the ``queued`` state without
    racing the worker.
    """
    job = InferenceJob(
        job_id=uuid.uuid4().hex,
        user=user,
        status=InferenceJob.STATUS_QUEUED,
        image_path=image_path,
        original_filename=original_filename,
    )
    job.save()

    _idle.clear()
    _queue.put(job.job_id)
    if start_worker:
        _ensure_worker(app)
    return job


def wait_for_idle(timeout=60.0):
    """Block until the queue has drained. Test-harness helper."""
    return _idle.wait(timeout=timeout)


def shutdown(timeout=5.0):
    """Stop the worker thread. Test-harness helper."""
    global _worker_thread
    _shutdown.set()
    _queue.put(None)
    thread = _worker_thread
    if thread is not None:
        thread.join(timeout=timeout)
    with _queue_lock:
        _worker_thread = None


def queue_depth():
    """Jobs waiting to be picked up. Diagnostics/tests."""
    return _queue.qsize()


# ── The job lifecycle ────────────────────────────────────────────────────────

def _mark(job, **fields):
    """Apply a state transition and persist it."""
    for key, value in fields.items():
        setattr(job, key, value)
    job.save()
    return job


def _fail(job, code, exc=None):
    """Terminate a job as failed, logging the real exception server-side.

    The stored message is the safe category text; ``exc`` never reaches Mongo.
    """
    if exc is not None:
        log.error("Inference job %s failed [%s]: %s", job.job_id, code, exc)
        log.debug("Traceback for job %s:\n%s", job.job_id,
                  traceback.format_exc())
    return _mark(
        job,
        status=InferenceJob.STATUS_FAILED,
        error_code=code,
        error_message=_SAFE_MESSAGES.get(code, _SAFE_MESSAGES[ERROR_INTERNAL]),
        completed_at=datetime.utcnow(),
    )


def _resolve_visitor(plate_text):
    """Visitor lookup + entry-time side effect.

    Mirrors ``process_plate`` exactly: same ``license_plate`` match, same
    authorized->status mapping, same first-detection entry timestamp, same
    fail-soft behaviour (a DB error must not lose the detection).
    """
    visitor = None
    status = "unknown"
    visitor_name = None

    if extensions.DB_ENABLED:
        try:
            visitor = Visitor.objects(license_plate=plate_text).first()
            status = "authorized" if visitor and visitor.authorized else "unauthorized"
            visitor_name = visitor.name if visitor else None

            if visitor and not visitor.entry_datetime_utc:
                try:
                    visitor.entry_datetime_utc = datetime.utcnow()
                    visitor.save()
                except Exception as upd_err:
                    print(f"Error setting entry time for {visitor.name}: {upd_err}")
        except Exception as db_err:
            print(f"Error querying visitor: {db_err}")

    return visitor, status, visitor_name


def _persist_detection(job, plate_text, plate_confidence, ocr_confidence,
                       status, visitor_name, user, image_path, processed_buffer):
    """Write the DetectionResult with the same semantics as ``process_plate``."""
    detection_timestamp = job.started_at or datetime.utcnow()

    original_path, processed_path = save_detection_images(
        _read_bytes(image_path), processed_buffer.tobytes()
    )

    record = DetectionResult(
        plate_number=plate_text,
        confidence=plate_confidence,
        ocr_confidence=ocr_confidence,
        status=status,
        visitor_name=visitor_name,
        processed_by=user,
        original_image_path=original_path,
        processed_image_path=processed_path,
        timestamp=detection_timestamp,
    )
    record.save()
    return record


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def _claim(job_id):
    """Atomically move a job from ``queued`` to ``processing``.

    This is a single ``findAndModify`` filtered on the current status, so only
    one runner can ever win the claim. A plain load-then-check would be a
    read-then-act race: two callers could both read ``queued`` and both go on
    to run inference, producing a duplicate DetectionResult.

    Returns the updated document, or ``None`` if the job is gone or already
    terminal (in which case it must not run again).
    """
    started = datetime.utcnow()
    return InferenceJob.objects(
        job_id=job_id,
        status=InferenceJob.STATUS_QUEUED,
    ).modify(
        new=True,
        status=InferenceJob.STATUS_PROCESSING,
        started_at=started,
        error_code=None,
        error_message=None,
        inc__attempts=1,
    )


def run_job(app, job_id):
    """Execute one job end to end. Never raises.

    queued -> processing -> completed | failed
    """
    with app.app_context():
        job = _claim(job_id)
        if job is None:
            existing = InferenceJob.objects(job_id=job_id).first()
            if existing is None:
                log.warning("Inference job %s vanished before it could run",
                            job_id)
            else:
                log.warning("Ignoring job %s already in state %s",
                            job_id, existing.status)
            return existing

        try:
            plate_detection_model = get_plate_detection_model()
            ocr_model = get_ocr_model()
            if plate_detection_model is None or ocr_model is None:
                return _fail(job, ERROR_INFERENCE)

            image_path = job.image_path
            if not image_path or not os.path.isfile(image_path):
                return _fail(job, ERROR_STORAGE)

            # The existing inference function, called unchanged.
            # char_details / error_msg are part of the returned contract; the
            # job document deliberately does not persist per-character detail.
            processed_img, plate_text, success, error_msg, plate_confidence, \
                ocr_confidence, _char_details = process_license_plate(
                    image_path, plate_detection_model, ocr_model
                )

            if not success:
                log.info("Inference job %s: no plate (%s)", job_id, error_msg)
                return _fail(job, ERROR_NO_PLATE)

            import cv2
            encoded, buffer = cv2.imencode(".jpg", processed_img)
            if not encoded:
                return _fail(job, ERROR_STORAGE)

            visitor, status, visitor_name = _resolve_visitor(plate_text)

            detection = None
            if extensions.DB_ENABLED:
                try:
                    detection = _persist_detection(
                        job, plate_text, plate_confidence, ocr_confidence,
                        status, visitor_name, job.user, image_path, buffer,
                    )
                except Exception as db_err:
                    # A detection that cannot be saved is a failed job, not a
                    # silent success with no result to reference.
                    return _fail(job, ERROR_PERSIST, db_err)

            return _complete(job, plate_text, detection, visitor)

        except Exception as exc:  # noqa: BLE001 - any failure must be terminal
            return _fail(job, ERROR_INTERNAL, exc)
        finally:
            _cleanup(job.image_path)


def _complete(job, plate_text, detection, visitor):
    """Terminal success transition."""
    return _mark(
        job,
        status=InferenceJob.STATUS_COMPLETED,
        completed_at=datetime.utcnow(),
        detection=detection,
        plate_text=plate_text,
    )


def _cleanup(image_path):
    """Remove the staged upload, mirroring process_plate's finally block."""
    try:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    except Exception as cleanup_err:
        log.warning("Error cleaning up temp file %s: %s", image_path, cleanup_err)

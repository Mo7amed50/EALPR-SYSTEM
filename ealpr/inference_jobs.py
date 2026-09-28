"""
inference_jobs.py — Phase 2 / Block C2 + Block C3.

Part 1 (C2): background inference worker.

WHY AN IN-PROCESS THREAD AND NOT CELERY
---------------------------------------
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


Part 2 (C3): job recovery and lifecycle hardening.

C2 made jobs durable but left three ways for a job to hang forever:

1. a job left ``queued`` when the process died — the Mongo document survived,
   but the in-memory ``queue.Queue`` did not, so nothing would ever run it;
2. a job left ``processing`` when the process died mid-inference — the claim had
   already been taken, so even a requeue would not re-run it, and the client
   could poll it indefinitely;
3. its staged upload in ``TEMP_DIR`` was never removed, because cleanup only ran
   from ``run_job``'s ``finally``.

``recover_jobs()`` closes all three. It is deliberately conservative:

* a still-plausible ``queued`` job is REQUEUED, not failed — the point of C3 is
  that legitimate work survives a restart, so valid work is not thrown away;
* a ``queued`` job too old to trust, and any ``processing`` job too old to
  still be running, are moved to ``failed`` with a stable error code. Stale
  processing jobs are never retried, because a retry would risk a duplicate
  DetectionResult for work that may already have been persisted;
* only files that can be *proven* to belong to an abandoned job are deleted.

Threshold rationale is documented at QUEUED_STALE_SECONDS / PROCESSING_STALE_
SECONDS below and derived from the timings measured in C2.
"""

import logging
import os
import queue
import re
import threading
import traceback
import uuid
from datetime import datetime, timedelta

from ealpr import extensions
from ealpr.ai_models import get_ocr_model, get_plate_detection_model
from ealpr.image_storage import save_detection_images
from ealpr.utils import process_license_plate, utc_to_cairo
from models import DetectionResult, InferenceJob, User, Visitor

log = logging.getLogger(__name__)


# Failure categories. These are stable, machine-readable and safe to expose.
ERROR_NO_PLATE = "no_plate_detected"
ERROR_INFERENCE = "inference_failed"
ERROR_STORAGE = "image_storage_failed"
ERROR_PERSIST = "detection_persist_failed"
ERROR_INTERNAL = "internal_error"
# Block C3: recovery terminations. Same naming convention as the C2 codes, and
# equally safe to expose - the stored text is fixed prose, never an exception.
#
# There is deliberately no "worker_recovery" code. Requeueing a job is not a
# failure, so the job stays clean and simply runs; only jobs recovery gives up
# on are given a code.
ERROR_STALE_QUEUED = "stale_queued"
ERROR_STALE_PROCESSING = "stale_processing"
ERROR_STARTUP_RECOVERY = "startup_recovery"

_RECOVERY_CODES = (ERROR_STALE_QUEUED, ERROR_STALE_PROCESSING)

# Safe, non-leaking client messages.
_SAFE_MESSAGES = {
    ERROR_NO_PLATE: "No license plate was detected in the uploaded image.",
    ERROR_INFERENCE: "The image could not be processed for plate recognition.",
    ERROR_STORAGE: "The uploaded image could not be read for storage.",
    ERROR_PERSIST: "The detection result could not be saved.",
    ERROR_INTERNAL: "The inference job failed unexpectedly.",
    ERROR_STALE_QUEUED: "The queued job was too old to run safely and was cancelled.",
    ERROR_STALE_PROCESSING: (
        "The job was interrupted and could not be resumed safely. "
        "Please submit the image again."
    ),
    ERROR_STARTUP_RECOVERY: "The job could not be recovered after a restart.",
}

# ── Block C3 thresholds ──────────────────────────────────────────────────────
#
# Derived from the C2 measurements, not picked round numbers:
#
#   cold POST /process_plate (one-time model load) : 8.526 s
#   warm POST /process_plate (steady state)        : 2.360 s
#   async submit -> terminal state                 : 3.892 s
#
# PROCESSING_STALE_SECONDS = 600 s
#   A processing job has already been claimed, and the POST endpoint refuses to
#   accept work until the models are loaded, so the *worker* normally pays only
#   the warm cost (~2.4 s). 600 s is ~150x the measured warm inference and ~60x
#   the measured end-to-end duration. That is deliberately far too generous to
#   kill a healthy job - including the first job after a restart, which does pay
#   the cold load - while still guaranteeing that a job orphaned by a dead
#   process becomes terminal instead of hanging. The requirement is to avoid
#   failing a healthy long-running job, so erring generous is the correct
#   direction.
#
# QUEUED_STALE_SECONDS = 1800 s
#   Queue wait is not compute time, so it needs a different and much looser
#   bound: at ~4 s per job a 30-minute wait is a backlog of roughly 450 jobs.
#   Past that point the job is not "waiting its turn", it is abandoned, and
#   holding its staged image on disk is pure waste. This is a trust boundary,
#   not a service-level target, which is why it is so much larger than the
#   processing bound.
#
# Both are env-overridable so an operator can widen them for slow hardware
# without a code change. Widening is always safe; the failure mode of a value
# that is too small is a spurious "failed" job, so these should only be lowered
# with evidence about real inference duration.
QUEUED_STALE_SECONDS = int(os.getenv("INFERENCE_QUEUED_STALE_SECONDS", 1800))
PROCESSING_STALE_SECONDS = int(os.getenv("INFERENCE_PROCESSING_STALE_SECONDS", 600))

# Prefix used by blueprints/inference.py when staging an upload. It is
# deliberately distinct from the "upload_<uuid>_" prefix used by
# blueprints/detection.py for /process_plate and /process_video, which is what
# makes artifact cleanup able to prove ownership: an "inference_job_" file is an
# async-job artifact, and "upload_" files belong to the synchronous endpoints
# and are never touched here.
STAGED_PREFIX = "inference_job_"

#: The exact shape blueprints/inference.py writes:
#:     "inference_job_{uuid4().hex}_{secure_filename}"
#:
#: Matching the whole shape rather than just the prefix is what makes artifact
#: cleanup safe to run against a shared TEMP_DIR. Only this module's own submit
#: path can produce a name that fits, so anything matching is an async-job
#: artifact by construction; a file that merely begins with the same letters is
#: not touched. Nothing else in the project stages uploads under TEMP_DIR with
#: this prefix - /process_plate and /process_video use "upload_".
_STAGED_ARTIFACT_RE = re.compile(r"^inference_job_[0-9a-f]{32}_.+$")


def is_staged_artifact(path):
    """True when ``path``'s basename is a well-formed async-job staged upload."""
    return bool(_STAGED_ARTIFACT_RE.match(os.path.basename(str(path))))

_queue = queue.Queue()
_queue_lock = threading.Lock()
_worker_thread = None
# Every consumer thread this process has ever started, and has not yet been seen
# to exit. C3.4 makes "at most one consumer" a structural invariant rather than
# a bookkeeping convention: _ensure_worker consults this registry instead of
# trusting a single variable, so a worker that outlives a failed shutdown() join
# can never be forgotten and then duplicated. Dead entries are purged lazily.
_all_workers = set()
_shutdown = threading.Event()
# Idle notification used by the test-suite to await deterministic drain.
_idle = threading.Event()
_idle.set()
# Block C3.4: ids currently sitting in the in-memory queue. Recovery re-queues by
# job_id, and without this a forced re-run of recovery (or a job enqueued twice)
# would push the same id in repeatedly. The atomic claim already makes duplicate
# execution impossible, so this is about not accumulating junk entries, not
# about correctness.
_queued_ids = set()
# Block C3.4: one-shot guard so create_app() can call the startup hook without
# ever producing a second consumer.
_initialization_done = False
_indexes_ensured = False



# ── Lifecycle ────────────────────────────────────────────────────────────────

def _ensure_worker(app):
    """Start the single consumer thread on first use.

    Idempotent under concurrency: the registry check and the thread creation both
    happen under ``_queue_lock``, so simultaneous callers cannot each start one.
    A thread that is still alive is reused even if ``_worker_thread`` no longer
    points at it, which is what stops a second consumer appearing after a
    ``shutdown()`` whose join timed out.
    """
    global _worker_thread
    with _queue_lock:
        for thread in [t for t in _all_workers if not t.is_alive()]:
            _all_workers.discard(thread)
        alive = [t for t in _all_workers if t.is_alive()]
        if alive:
            _worker_thread = alive[0]
            return _worker_thread
        _shutdown.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(app,),
            daemon=True,
            name="inference-worker",
        )
        _all_workers.add(_worker_thread)
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
            with _queue_lock:
                _queued_ids.discard(job_id)
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
    with _queue_lock:
        _queued_ids.add(job.job_id)
    _queue.put(job.job_id)
    if start_worker:
        _ensure_worker(app)
    return job


def queue_depth():
    """Jobs waiting to be picked up. Diagnostics/tests."""
    return _queue.qsize()


def queued_ids():
    """A snapshot of the ids currently held in the in-memory queue."""
    with _queue_lock:
        return set(_queued_ids)


def drain_queue():
    """Discard everything waiting in memory, keeping ``_queued_ids`` in step.

    Used by ``shutdown()`` and by the test harness. Without keeping the set in
    step, a discarded id would stay marked as queued and recovery would
    permanently refuse to re-queue it.
    """
    while True:
        try:
            job_id = _queue.get_nowait()
        except queue.Empty:
            break
        with _queue_lock:
            _queued_ids.discard(job_id)
        _queue.task_done()
    _idle.set()


def wait_for_idle(timeout=60.0):
    """Block until the queue has drained. Test-harness helper."""
    return _idle.wait(timeout=timeout)


def shutdown(timeout=5.0):
    """Stop the worker thread and drop anything still queued. Test-harness helper.

    If the join times out the thread is *left registered* rather than forgotten,
    so ``_ensure_worker`` will reuse it instead of starting a second consumer. A
    hard stop is therefore never silently traded against the one-consumer
    invariant.

    C3.4 note: the *initialisation* guard is deliberately not reset here. This
    models stopping a consumer, not a process restart, and re-running recovery
    is what ``initialize_inference_jobs(force=True)`` is for.
    """
    global _worker_thread
    _shutdown.set()
    _queue.put(None)
    with _queue_lock:
        thread = _worker_thread
    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            log.warning("Inference worker did not stop within %.1fs; leaving it "
                        "registered so no second consumer is created", timeout)
            return False
    with _queue_lock:
        _worker_thread = None
        for dead in [t for t in _all_workers if not t.is_alive()]:
            _all_workers.discard(dead)
    drain_queue()
    return True



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


# ── Block C3: startup recovery ───────────────────────────────────────────────

def _ensure_retention_index():
    """Create the InferenceJob indexes, including the C3 TTL retention index.

    Idempotent and fail-soft: MongoDB already indexes a collection's existing
    definitions, so this is a single cheap ``createIndexes`` on first start and a
    no-op assertion thereafter. A failure here costs retention, not
    availability, so it is logged and swallowed - the application must still
    start, and a missing TTL index is far less harmful than a failed boot.
    """
    global _indexes_ensured
    with _queue_lock:
        if _indexes_ensured:
            return True
    try:
        InferenceJob.ensure_indexes()
        with _queue_lock:
            _indexes_ensured = True
        log.info("InferenceJob indexes ensured (TTL retention on completed_at)")
        return True
    except Exception as index_err:
        log.warning("Could not ensure InferenceJob indexes: %s", index_err)
        return False


def recover_staged_artifact(image_path):
    """Delete a staged upload only when it is provably an async-job artifact.

    Two conditions must both hold, and this function refuses to guess:

    1. the file lives in ``TEMP_DIR`` and its basename carries the
       ``inference_job_`` prefix, which is the prefix
       blueprints/inference.py uses and which does not collide with the
       ``upload_`` prefix used by the synchronous endpoints; and
    2. the caller has already established the owning job is being abandoned.

    Age alone is never sufficient and no directory listing is ever swept, so
    unrelated temporary files cannot be caught by it. A missing file is not an
    error - the job may already have cleaned up after itself.
    """
    if not image_path:
        return False
    try:
        name = os.path.basename(image_path)
        if not is_staged_artifact(name):
            log.debug("Refusing to clean %r: not an inference-job artifact", name)
            return False
        if not os.path.exists(image_path):
            return False
        os.remove(image_path)
        log.info("Removed abandoned staged inference artifact (%d-char name)",
                 len(name))
        return True
    except Exception as cleanup_err:
        # C3.10: a permission error or a racing deletion must never abort
        # recovery for the other jobs.
        log.warning("Could not remove staged artifact %r: %s",
                    os.path.basename(str(image_path)), cleanup_err)
        return False


def _fail_stale(build_query, code, now, counts, label):
    """Atomically terminate a set of stale jobs and free their staged files.

    ``build_query`` is a callable returning a fresh queryset, because a MongoEngine
    queryset cannot be iterated and then reused for a write.

    The transition is a single conditional bulk ``update`` filtered on the job's
    *current* status and age. That filter is the safety property: a job that a
    live worker claimed between the candidate scan and this call is no longer
    ``queued``/``processing`` under the stale predicate, so it cannot be failed
    out from under itself. The read-back is the other half of that guarantee -
    only jobs that really did transition are reported, so the returned count and
    the reclaimed files can never disagree with the database.

    The staged paths are re-read *after* the transition rather than taken from
    the candidate scan. That ordering is deliberate. If recovery had acted on the
    pre-transition list it could delete a file that the worker had just claimed
    and was about to read, turning a healthy job into ``image_storage_failed``.
    Reading back only the jobs that are now ``failed`` with a recovery code
    makes the persisted state the single source of truth.
    """
    candidates = [j.job_id for j in build_query().only("job_id")]
    if not candidates:
        return 0

    # update(multi=True), NOT modify(). modify() is a find_one_and_update and
    # therefore transitions only a SINGLE document - correct for _claim, where
    # exactly one runner must win, but wrong here: it would fail just one stale
    # job per startup and leave the rest hanging, which is the very bug this
    # block exists to remove. The status filter keeps the multi-update
    # conditional, so a job claimed by a live worker mid-sweep is still spared.
    build_query().update(
        status=InferenceJob.STATUS_FAILED,
        error_code=code,
        error_message=_SAFE_MESSAGES[code],
        completed_at=now,
    )

    # Authoritative: only jobs that really did transition.
    transitioned = list(
        InferenceJob.objects(
            job_id__in=candidates,
            status=InferenceJob.STATUS_FAILED,
            error_code=code,
        ).only("image_path")
    )
    for job in transitioned:
        recover_staged_artifact(job.image_path)

    counts[label] = len(transitioned)
    log.info("Recovery failed %d stale %s job(s) [%s]", len(transitioned), label, code)
    return len(transitioned)


def recover_jobs(now=None):
    """Reconcile persisted job state with the (empty) in-memory queue.

    This models what the process *cannot* know: a restart loses the queue, so
    MongoDB is the only record of what was in flight. The policy is:

    ==================  ==========================================  ===========
    Persisted state     Action                                      Rationale
    ==================  ==========================================  ===========
    queued, recent      re-queue, then run normally                 valid work
    queued, too old     fail ``stale_queued`` + free its file        untrustworthy
    processing, too old fail ``stale_processing`` + free its file    never retried
    completed / failed  untouched                                   already terminal
    ==================  ==========================================  ===========

    A stale ``processing`` job is failed rather than retried on purpose. Its
    claim is long gone, but the process may have died *after* persisting a
    DetectionResult and *before* marking the job complete. Re-running would
    create a second detection for one image. Failing is recoverable by the user;
    a silent duplicate is not.

    Never raises: every step is guarded, one unusable job cannot stop the rest,
    and a MongoDB outage degrades to "recovery skipped" rather than a failed
    boot.
    """
    now = now or datetime.utcnow()
    counts = {
        "requeued": 0, "skipped_already_queued": 0,
        "stale_queued": 0, "stale_processing": 0, "unrunnable_queued": 0,
        "orphan_artifacts": 0, "errors": 0,
    }

    try:
        _ensure_retention_index()

        if not extensions.DB_ENABLED:
            log.info("Database disabled; skipping inference job recovery.")
            return counts

        queued_cutoff = now - timedelta(seconds=QUEUED_STALE_SECONDS)
        processing_cutoff = now - timedelta(seconds=PROCESSING_STALE_SECONDS)

        # --- queued: split into runnable, unrunnable and abandoned -----------
        # A queued job is requeued only if it is young enough that a backlog
        # could still explain the wait *and* it is structurally able to finish.
        # The second condition is not theoretical: _persist_detection stores
        # `processed_by=job.user`, so a job whose user document has been deleted
        # can never be persisted. Requeuing one would burn a full YOLO/OCR
        # inference and then fail with detection_persist_failed, wasting the
        # scarce serialised resource on a job with no possible outcome.
        raw_queued = list(InferenceJob._get_collection().find(
            {
                "status": InferenceJob.STATUS_QUEUED,
                "created_at": {"$gte": queued_cutoff},
            },
            {"job_id": 1, "user": 1},
        ))
        referenced = {d["user"] for d in raw_queued if d.get("user")}
        existing_users = set()
        if referenced:
            existing_users = {
                u["_id"] for u in User._get_collection().find(
                    {"_id": {"$in": list(referenced)}}, {"_id": 1})
            }
        runnable, unrunnable = [], []
        for doc in raw_queued:
            owner = doc.get("user")
            # A job with no owner at all is still runnable: processed_by is
            # simply left empty. Only a *dangling* reference is unrunnable.
            if owner is None or owner in existing_users:
                runnable.append(doc["job_id"])
            else:
                unrunnable.append(doc["job_id"])

        if unrunnable:
            _fail_stale(
                lambda: InferenceJob.objects(
                    job_id__in=unrunnable,
                    status=InferenceJob.STATUS_QUEUED,
                ),
                ERROR_STARTUP_RECOVERY, now, counts, "unrunnable_queued",
            )

        already_queued = queued_ids()
        for job_id in runnable:
            if job_id in already_queued:
                counts["skipped_already_queued"] += 1
                continue
            _idle.clear()
            with _queue_lock:
                _queued_ids.add(job_id)
            _queue.put(job_id)
            counts["requeued"] += 1

        stale_queued = lambda: InferenceJob.objects(  # noqa: E731
            status=InferenceJob.STATUS_QUEUED,
            created_at__lt=queued_cutoff,
        )
        _fail_stale(stale_queued, ERROR_STALE_QUEUED, now, counts, "stale_queued")

        # --- processing: only the clearly-dead ones --------------------------
        # started_at is set by the atomic claim, so it is the true time work
        # began. created_at is used only as a defensive fallback for a document
        # that somehow reached "processing" without a stamp.
        def stale_processing():
            return InferenceJob.objects(
                status=InferenceJob.STATUS_PROCESSING,
            ).filter(
                __raw__={
                    "$or": [
                        {"started_at": {"$lt": processing_cutoff}},
                        {"started_at": None, "created_at": {"$lt": processing_cutoff}},
                    ]
                }
            )
        _fail_stale(stale_processing, ERROR_STALE_PROCESSING, now, counts,
                    "stale_processing")

        counts["orphan_artifacts"] = cleanup_orphan_artifacts(
            older_than_seconds=QUEUED_STALE_SECONDS)

    except Exception as recovery_err:  # noqa: BLE001 - startup must survive
        counts["errors"] += 1
        log.error("Inference job recovery could not complete: %s", recovery_err)
        log.debug("Recovery traceback:\n%s", traceback.format_exc())

    if any(counts[k] for k in counts if k != "errors"):
        log.info("Inference job recovery summary: %s", counts)
    return counts


def cleanup_orphan_artifacts(older_than_seconds=None):
    """Remove staged ``inference_job_`` files that no job document claims.

    This is the only filesystem sweep in the block, and it is ownership-based
    rather than age-based. A file qualifies only if BOTH hold:

    * it carries the ``inference_job_`` prefix written by the C2/C3 submit path;
      "upload_" files from /process_plate and /process_video, and every other
      unrelated temporary file, are excluded by name; and
    * no ``InferenceJob`` document records it as ``image_path``, i.e. the job
      that owned it is gone, so the file cannot be in use.

    Age is applied only as a safety margin on top of those two conditions, never
    as the sole criterion. A file staged microseconds ago by a request that has
    not yet called ``enqueue()`` is briefly ownerless; the margin keeps that
    window from ever being destructive.

    Returns the number of files removed. Never raises.
    """
    if not extensions.DB_ENABLED:
        return 0
    if older_than_seconds is None:
        older_than_seconds = QUEUED_STALE_SECONDS
    removed = 0
    try:
        from config import TEMP_DIR

        if not TEMP_DIR.exists():
            return 0
        cutoff = datetime.utcnow() - timedelta(seconds=older_than_seconds)
        claimed = {
            p for p in InferenceJob.objects(
                image_path__ne=None, image_path__nin=[""]
            ).only("image_path").values_list("image_path")
            if p
        }
        for entry in TEMP_DIR.iterdir():
            if not entry.is_file():
                continue
            if not is_staged_artifact(entry.name):
                continue
            if str(entry) in claimed:
                continue
            try:
                mtime = datetime.utcfromtimestamp(entry.stat().st_mtime)
            except OSError:
                continue
            if mtime >= cutoff:
                continue
            if recover_staged_artifact(str(entry)):
                removed += 1
    except Exception as cleanup_err:
        # C3.10: never let a filesystem problem abort startup.
        log.warning("Orphan artifact cleanup could not complete: %s", cleanup_err)
    if removed:
        log.info("Removed %d orphaned staged inference artifact(s)", removed)
    return removed


def initialize_inference_jobs(app, force=False):
    """Block C3.4 idempotent startup hook. Safe to call any number of times.

    Guarantees, in order:

    1. at most one consumer thread, ever - the guard and ``_ensure_worker`` are
       both taken under ``_queue_lock``, so concurrent first callers cannot each
       start one;
    2. recovery runs at most once per process unless ``force=True``, which is how
       the test-suite simulates a restart within one interpreter;
    3. a failure anywhere is contained - ``recover_jobs`` never raises, and this
       function never raises either, because a recovery problem must not stop the
       application from serving requests.

    The consumer is only started when recovery actually produced work, or when
    one is already needed. That preserves C2's lazy-start behaviour instead of
    spawning a thread on every ``create_app()``.
    """
    global _initialization_done

    with _queue_lock:
        if _initialization_done and not force:
            return {"skipped": True}
        _initialization_done = True

    counts = {"skipped": False}
    try:
        with app.app_context():
            counts = recover_jobs()
    except Exception as init_err:  # noqa: BLE001 - belt and braces
        log.error("Inference job startup initialisation failed: %s", init_err)
        counts = {"error": True}

    if counts.get("requeued"):
        try:
            _ensure_worker(app)
        except Exception as worker_err:
            # Work is durable and re-queued; the next submit restarts the
            # consumer, so this is recoverable rather than fatal.
            log.error("Could not start the inference worker during recovery: %s",
                      worker_err)
    return counts


def worker_is_running():
    """True when a live consumer thread exists. Diagnostics/tests."""
    thread = _worker_thread
    return thread is not None and thread.is_alive()


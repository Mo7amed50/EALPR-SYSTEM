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


Part 3 (C4): worker health monitoring and supervision.

The C4 audit established that the consumer thread is very unlikely to die from an
uncaught exception, so the real reliability gap is not crashes - it is SILENT
STRANDING: work that reaches a non-terminal Mongo state with no consumer and no
mechanism to resolve it, while the process keeps serving traffic and every signal
reads normal. Two concrete holes were found:

1. A job left ``processing`` less than PROCESSING_STALE_SECONDS before a restart
   matched neither branch of the stale-processing predicate, so recovery skipped it
   and it stayed non-terminal until some later restart at least that far away.
   After a restart there is by definition no consumer that could own such a job, so
   its age is irrelevant. ``recover_jobs(orphan_processing=True)`` closes this, and
   it FAILS such a job - it never requeues it, because the DetectionResult insert
   and the job completion are two separate writes with no transaction between them.

2. Nothing observed worker liveness. ``worker_is_running()`` already existed but had
   no production caller, so a dead consumer with a backlog was invisible until the
   next submission. The monitor below gives it one.

The monitor calls ``_ensure_worker`` and NEVER creates a thread itself, so
"at most one consumer" remains structural rather than conventional. It also never
treats "no worker" as a fault on its own: the architecture lazy-starts, so
"no work + no worker" is a valid, healthy state that must not trigger a restart.
"""

import logging
import os
import queue
import re
import threading
import time
import traceback
import uuid
from collections import deque
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

# ── Block C4: worker supervision ──────────────────────────────────────────────
#
# The monitor is deliberately a *supervisor*, not a babysitter. Its single job is
# to notice the one bad state - work outstanding with no live consumer - and to
# call the existing _ensure_worker. It never creates a thread itself, so C3's
# "at most one consumer" invariant is preserved by construction rather than by
# convention, and it never runs inference itself.
#
# The tick is a pure in-memory read (queue depth plus the _queued_ids mirror), so
# monitoring adds NO MongoDB round-trip. That matters: Atlas latency dominates
# everything here, and a monitor that queried InferenceJob each tick would add a
# network call to a subsystem whose measured warm inference is ~2.4s.
MONITOR_INTERVAL_SECONDS = float(
    os.getenv("INFERENCE_MONITOR_INTERVAL_SECONDS", 5)
)

# Restart rate limiting. Without this, a consumer that dies immediately after every
# start would be restarted forever: unbounded thread creation, unbounded CPU, and
# each new thread pushing an app_context. C4 caps attempts per window, so a
# persistent fault degrades to "no consumer" - which the operator can see in the
# logs - instead of a resource-exhaustion loop.
#
# The default of 5 attempts per 5 minutes is generous for the realistic case (one
# unlucky death) and still bounded for the pathological one. Widening is always
# safe; narrowing can only cause a real recovery to be suppressed.
MAX_WORKER_RESTARTS = int(os.getenv("INFERENCE_MAX_WORKER_RESTARTS", 5))
WORKER_RESTART_WINDOW_SECONDS = float(
    os.getenv("INFERENCE_WORKER_RESTART_WINDOW_SECONDS", 300)
)

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
#
# Block C4.2: this is no longer set unconditionally. It is claimed only once
# recovery has actually SUCCEEDED, so a MongoDB outage at startup does not
# permanently disable recovery for the life of the process - the monitor retries
# until it succeeds. See initialize_inference_jobs.
_initialization_done = False
_indexes_ensured = False

# ── Block C4: monitor state ───────────────────────────────────────────────────
# Kept deliberately small: one event, one thread reference, a sliding window of
# restart timestamps, and two counters. The monitor is NOT added to _all_workers,
# because that registry is the single-consumer authority and the monitor is not
# a consumer - mixing them would make worker_is_running() report on the wrong
# thread.
_monitor_stop = threading.Event()
_monitor_thread = None
_monitor_lock = threading.Lock()
_restart_times = deque()
_restarts_total = 0
_restart_suppressed = False



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
        # No image path, no plate text, no exception payload - just the fact that a
        # consumer now exists, matching C3's logging discipline.
        log.info("Inference worker started")
        return _worker_thread


def _worker_loop(app):
    """Consume jobs one at a time, forever."""
    try:
        _worker_loop_body(app)
    finally:
        # Reached on clean shutdown (_shutdown set) and on any escape that
        # terminates the thread. Either way the fact is worth recording, because
        # "the consumer went away" is the event C4 supervision exists to notice.
        log.info("Inference worker exited")


def _worker_loop_body(app):
    """The consume loop, split out so _worker_loop can log its exit."""
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


# ── Block C4: worker health fact ──────────────────────────────────────────────

def worker_health():
    """One snapshot of everything the supervisor needs to decide. Read-only.

    Built entirely from in-memory primitives - the queue, the ``_queued_ids``
    mirror, and thread liveness - so calling it costs no MongoDB round-trip and
    can safely run on every monitor tick.

    ``work_outstanding`` is the whole point of this function, and it is
    deliberately NOT "worker_running". The architecture lazy-starts: an empty
    queue with no consumer is a valid, healthy state, and treating it as a fault
    would start a thread that has nothing to do. Supervision must act on
    *outstanding work*, never on the absence of a thread by itself.

    Read each lock separately and never hold two at once, so this can never
    deadlock against _ensure_worker or the monitor.
    """
    depth = _queue.qsize()
    with _queue_lock:
        tracked = len(_queued_ids)
        recovery_completed = _initialization_done
    with _monitor_lock:
        restarts = _restarts_total
        suppressed = _restart_suppressed
    monitor = _monitor_thread
    thread = _worker_thread
    return {
        "worker_running": thread is not None and thread.is_alive(),
        "queue_depth": depth,
        "outstanding_ids": max(depth, tracked),
        "work_outstanding": depth > 0 or tracked > 0,
        "recovery_completed": recovery_completed,
        "monitor_running": monitor is not None and monitor.is_alive(),
        "restarts": restarts,
        "restart_suppressed": suppressed,
    }


# ── Block C4: the monitor ─────────────────────────────────────────────────────

def _consume_restart_budget():
    """Record and authorise one restart attempt. Returns False when capped.

    Separated from the restart itself because ``_ensure_worker`` takes
    ``_queue_lock`` and this takes ``_monitor_lock``; keeping the decision and
    the action in different lock scopes means the monitor can never deadlock
    against the worker creator, whichever order they are taken in.
    """
    now = time.monotonic()
    global _restart_suppressed, _restarts_total
    with _monitor_lock:
        while _restart_times and (now - _restart_times[0]) > WORKER_RESTART_WINDOW_SECONDS:
            _restart_times.popleft()
        if len(_restart_times) >= MAX_WORKER_RESTARTS:
            if not _restart_suppressed:
                _restart_suppressed = True
                log.warning(
                    "Inference worker restart cap reached: %d attempts in %.0fs; "
                    "further restarts suppressed until the window expires. "
                    "Outstanding work will wait for a process restart.",
                    MAX_WORKER_RESTARTS, WORKER_RESTART_WINDOW_SECONDS)
            return False
        _restart_suppressed = False
        _restart_times.append(now)
        _restarts_total += 1
        return True


def _restart_worker(app, reason):
    """Restart the consumer by delegating to the one legal creation path."""
    if not _consume_restart_budget():
        return False
    log.info("Inference worker restart requested (%s)", reason)
    _ensure_worker(app)
    log.info("Inference worker restart performed (%s); restarts this process=%d",
             reason, worker_health()["restarts"])
    return True


def _retry_recovery(app):
    """C4.2 — re-attempt a recovery that previously failed.

    ``initialize_inference_jobs`` only claims its one-shot guard once recovery has
    actually succeeded, so after a startup MongoDB outage the guard stays unset
    and this is the path that eventually converges. It stops for good the first
    time recovery succeeds; if MongoDB is permanently down this is one failed
    attempt per tick, which is self-limiting rather than a busy loop.
    """
    global _initialization_done
    with _queue_lock:
        if _initialization_done:
            return False
    try:
        with app.app_context():
            counts = recover_jobs(orphan_processing=not worker_is_running())
        if counts.get("errors"):
            return False
    except Exception as recovery_err:  # noqa: BLE001 - never kill the monitor
        log.warning("Inference job recovery retry failed: %s", recovery_err)
        return False
    with _queue_lock:
        _initialization_done = True
    log.info("Inference job recovery retry succeeded: %s", counts)
    return True


def _monitor_tick(app):
    """One supervision decision. Small and side-effect-obvious on purpose."""
    _retry_recovery(app)
    health = worker_health()
    # Both conditions matter. "No worker" alone is the lazy-start state and must
    # be left alone; "work outstanding" alone is being handled fine.
    if not health["work_outstanding"] or health["worker_running"]:
        return False
    return _restart_worker(app, "{} job(s) outstanding with no live consumer"
                           .format(health["outstanding_ids"]))


def _monitor_loop(app):
    log.info("Inference worker monitor started (interval %.1fs)",
             MONITOR_INTERVAL_SECONDS)
    try:
        while not _monitor_stop.wait(MONITOR_INTERVAL_SECONDS):
            try:
                _monitor_tick(app)
            except Exception:  # noqa: BLE001 - the monitor must outlive any tick
                log.exception("Inference worker monitor tick failed")
    finally:
        log.info("Inference worker monitor exited")


def _ensure_monitor(app):
    """Start the single monitor thread. Idempotent; never the worker's business."""
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return _monitor_thread
        _monitor_stop.clear()
        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(app,),
            daemon=True,
            name="inference-monitor",
        )
        _monitor_thread.start()
        return _monitor_thread


def _stop_monitor(timeout=5.0):
    """Stop the monitor and reset restart accounting. Test/lifecycle helper.

    Restart bookkeeping is reset here so a later supervision episode is not
    penalised by a previous one's attempts.
    """
    global _monitor_thread, _restarts_total, _restart_suppressed
    _monitor_stop.set()
    with _monitor_lock:
        monitor = _monitor_thread
    if monitor is not None:
        monitor.join(timeout=timeout)
        if monitor.is_alive():
            log.warning("Inference worker monitor did not stop within %.1fs",
                        timeout)
            return False
    with _monitor_lock:
        _monitor_thread = None
        _restart_times.clear()
        _restarts_total = 0
        _restart_suppressed = False
    return True


def monitor_is_running():
    """True when a live monitor thread exists. Diagnostics/tests."""
    monitor = _monitor_thread
    return monitor is not None and monitor.is_alive()


def shutdown(timeout=5.0):
    """Stop the consumer and the monitor, then drop anything still queued.

    Test-harness and lifecycle helper.

    If the join times out the thread is *left registered* rather than forgotten,
    so ``_ensure_worker`` will reuse it instead of starting a second consumer. A
    hard stop is therefore never silently traded against the one-consumer
    invariant.

    The monitor is stopped first. Leaving it running across a consumer shutdown
    would immediately satisfy its own condition - work outstanding, no consumer -
    and race the harness by restarting the thread the test just stopped. The
    restart budget is reset with it, so a later episode starts clean.

    C3.4 note: the *initialisation* guard is deliberately not reset here. This
    models stopping a consumer, not a process restart, and re-running recovery
    is what ``initialize_inference_jobs(force=True)`` is for.
    """
    global _worker_thread
    _stop_monitor(timeout=timeout)
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


def recover_jobs(now=None, orphan_processing=False):
    """Reconcile persisted job state with the (empty) in-memory queue.

    This models what the process *cannot* know: a restart loses the queue, so
    MongoDB is the only record of what was in flight. The policy is:

    ==================  ==========================================  ===========
    Persisted state     Action                                      Rationale
    ==================  ==========================================  ===========
    queued, recent      re-queue, then run normally                 valid work
    queued, too old     fail ``stale_queued`` + free its file        untrustworthy
    processing, too old fail ``stale_processing`` + free its file    never retried
    processing, no live fail ``stale_processing`` + free its file    C4.1
      consumer (C4.1)
    completed / failed  untouched                                   already terminal
    ==================  ==========================================  ===========

    A stale ``processing`` job is failed rather than retried on purpose. Its
    claim is long gone, but the process may have died *after* persisting a
    DetectionResult and *before* marking the job complete. Re-running would
    create a second detection for one image. Failing is recoverable by the user;
    a silent duplicate is not.

    ``orphan_processing`` is the C4.1 fix for a hole the age rule cannot reach: a
    job interrupted by a crash and restarted seconds later is not yet "too old",
    so it was skipped and left non-terminal. It defaults to **False**, so every
    C3 caller keeps exactly the C3 behaviour; only startup recovery, which knows
    no consumer can be running, opts in. It is a no-op whenever a live consumer
    does exist, so it can never fail a job that is genuinely being worked on.
    New count key: ``orphaned_processing``, reported separately from the
    age-based ``stale_processing`` so the two causes stay distinguishable.

    Never raises: every step is guarded, one unusable job cannot stop the rest,
    and a MongoDB outage degrades to "recovery skipped" rather than a failed
    boot.
    """
    now = now or datetime.utcnow()
    counts = {
        "requeued": 0, "skipped_already_queued": 0,
        "stale_queued": 0, "stale_processing": 0, "unrunnable_queued": 0,
        "orphaned_processing": 0,
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

        # --- processing: abandoned work first, then age ----------------------
        # C4.1. The age-based rule below can only see a `processing` job once it
        # is older than PROCESSING_STALE_SECONDS, which left a real hole: a job
        # interrupted by a crash and restarted 10 seconds later is *not* stale,
        # so recovery skipped it, and it stayed non-terminal until some later
        # restart at least 600s afterwards. A crash-and-immediately-restart is
        # the single most common deployment event and it lands squarely in that
        # window.
        #
        # The fix is not a smaller threshold - that would be a new, evidence-free
        # constant. It is to notice the thing that is actually decisive at
        # startup: with no live consumer there is, by definition, nothing that
        # could still be running the job, so its age is irrelevant.
        #
        # It FAILS the job. It never requeues it. Requeueing is the one action
        # that could duplicate a DetectionResult, because _persist_detection and
        # _complete are two separate writes with no transaction between them.
        if orphan_processing and not worker_is_running():
            counts["orphaned_processing"] = _fail_stale(
                lambda: InferenceJob.objects(
                    status=InferenceJob.STATUS_PROCESSING,
                ),
                ERROR_STALE_PROCESSING, now, counts, "orphaned_processing",
            )

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
    """Block C3.4/C4 startup hook. Safe to call any number of times.

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

    C4.2 - the guard is now claimed only on SUCCESS. C3 set it *before* running
    recovery, so a MongoDB outage at startup logged one error and then disabled
    recovery for the life of the process: jobs persisted before the outage could
    never be re-queued or failed, silently, until someone restarted the process
    by hand. Leaving the guard unset on failure lets the monitor retry until
    MongoDB answers, and retrying is safe because recovery is idempotent.
    """
    global _initialization_done

    with _queue_lock:
        if _initialization_done and not force:
            return {"skipped": True}

    counts = {"skipped": False}
    recovered = False
    try:
        with app.app_context():
            # C4.1: at startup there is no live consumer, so any `processing`
            # job found here is abandoned regardless of age. Guarded on
            # worker_is_running() inside recover_jobs, so a forced re-run while
            # a consumer is live still cannot disturb a real job.
            counts = recover_jobs(orphan_processing=not worker_is_running())
        recovered = not counts.get("errors")
    except Exception as init_err:  # noqa: BLE001 - belt and braces
        log.error("Inference job startup initialisation failed: %s", init_err)
        counts = {"error": True}

    if recovered:
        with _queue_lock:
            _initialization_done = True
    else:
        log.warning(
            "Inference job recovery did not complete; it will be retried by the "
            "worker monitor until it succeeds")

    # C4.4: the monitor is started unconditionally, not only when work was
    # recovered. It is the thing that repairs both holes C4 is here for - a
    # failed recovery and a consumer that goes missing - so gating it on
    # counts["requeued"] would disable supervision exactly when it is needed.
    try:
        _ensure_monitor(app)
    except Exception as monitor_err:
        log.error("Could not start the inference worker monitor: %s", monitor_err)

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


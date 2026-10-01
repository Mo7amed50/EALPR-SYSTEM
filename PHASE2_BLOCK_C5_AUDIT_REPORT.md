# PHASE 2 — BLOCK C5 AUDIT REPORT

## Production Runtime Hardening — AUDIT ONLY

> **NO APPLICATION CODE WAS MODIFIED.**
> No configuration was changed, no dependency was installed, no commit was
> created. The only file this block created is this report.

---

## 1. Executive Summary

The EALPR system is currently a **single-process Flask application running on the
Werkzeug development server with `debug=True`**, listening on `0.0.0.0:5000`.
There is no production server, no WSGI entry point, no deployment artifact, and
no shutdown lifecycle handling anywhere in the repository.

The C4 baseline is sound and is **not** the source of any high-severity finding.
The severity in this block sits entirely in the *process* layer that was never
built: the runtime entry point, the absence of configuration separation, the
absence of shutdown handling, and the absence of any statement about the
single-process requirement that C4 fundamentally depends on.

### Headline findings

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | `debug=True` + `host=0.0.0.0` activates the Werkzeug interactive debugger (code execution) on every network interface | **Critical** | Verified |
| 2 | `allow_unsafe_werkzeug=True` deliberately bypasses Flask-SocketIO's own "not for production" guard | **High** | Verified |
| 3 | No production WSGI server, no `wsgi.py`, no Dockerfile/Procfile/systemd unit, no server in `requirements.txt` | **High** | Verified |
| 4 | Zero shutdown lifecycle handling: no `atexit`, no `SIGTERM`/`SIGINT`, no `teardown_appcontext` | **High** | Verified |
| 5 | `use_reloader=False` is the *only* thing preventing a second process on every code edit; `use_reloader` defaults to `debug` | **High** | Verified |
| 6 | Multi-process would let one process's startup recovery **fail a job another process is actively processing** | **High** | Inferred |
| 7 | Multi-process would strand `queued` jobs forever (supervision is process-local) | **High** | Inferred |
| 8 | No production/development configuration separation exists at all | Medium | Verified |
| 9 | Default admin password `admin123` is not overridden; code only prints a warning | Medium | Verified |
| 10 | Socket.IO has no message queue, so multi-process would silently break live broadcasts | Medium | Verified |
| 11 | `.env.example` documents `retryWrites=true&w=majority`, which `create_app` explicitly overrides to `False` | Medium | Verified |
| 12 | C4's `shutdown()` is never called by the application — it is test-harness-only | Medium | Verified |

### The one piece of good news

MongoDB's atomic `_claim` (`ealpr/inference_jobs.py:714-736`) is a single
conditional `findAndModify` filtered on `status="queued"`, and `job_id` carries a
unique index. **This does prevent duplicate inference across processes.** The
multi-process risk in this system is stranded and incorrectly-failed jobs, not
duplicate detections. That materially lowers the worst case and should be stated
plainly rather than alarmingly.

---

## 2. Baseline / Commit

| Item | Value |
|---|---|
| Repository | `D:\EALPR SYSTEM` |
| Branch | `main` |
| Audit baseline commit | `c1814880` — `phase2: harden worker supervision and recovery` |
| Working tree at audit start | Clean (only tracked `.pyc` byte-changes) |
| Working tree at audit end | Clean — **no file modified except this report** |

C4 is treated as the immutable baseline. This audit proposes no change to any C4
mechanism.

---

## 3. Repository Runtime Architecture

```
python app.py                          <-- the only documented run command
  │
  ├─ app.py:10        app = create_app()        (executes at IMPORT time)
  │    └─ ealpr/__init__.py:22  create_app()
  │         ├─ :24       load_dotenv()
  │         ├─ :26-31    Flask(...)             template/static folders
  │         ├─ :32       app.config["SECRET_KEY"]
  │         ├─ :35       os.makedirs(TEMP_DIR)
  │         ├─ :38-41    login_manager.init_app
  │         ├─ :43       socketio.init_app(async_mode="threading")
  │         ├─ :46       _connect_mongodb()     3 attempts, 5s timeouts
  │         ├─ :49-69    user_loader            (offline-mode fallback)
  │         ├─ :71-113   4 template filters/context processors
  │         ├─ :116-133  8 blueprints registered
  │         ├─ :136      _create_default_admin()
  │         └─ :144-148  initialize_inference_jobs(app)   <-- C2/C3/C4 hook
  │              └─ recover_jobs(orphan_processing=…)  +  _ensure_monitor()
  │
  └─ app.py:13-19     socketio.run(host="0.0.0.0", port=5000,
                                     debug=True, use_reloader=False,
                                     allow_unsafe_werkzeug=True)
         └─ flask_socketio/__init__.py:606-661
              └─ app.run(threaded=True, use_reloader=False)
                   └─ werkzeug.serving.run_simple  <-- DEV SERVER
```

### Process-local state (the C4 assumption surface)

| Object | Location | Scope |
|---|---|---|
| `_queue` (`queue.Queue`) | `inference_jobs.py` | **per process** |
| `_queued_ids` (set) | `inference_jobs.py` | **per process** |
| `_all_workers` (set) | `inference_jobs.py` | **per process** |
| `_monitor_thread`, `_restart_times`, counters | `inference_jobs.py:267-272` | **per process** |
| `_initialization_done` | `inference_jobs.py:258` | **per process** |
| YOLO / OCR weights | `ealpr/ai_models.py` | per process (in-memory) |
| `socketio` client registry | python-socketio `Server` | **per process** |
| `DB_ENABLED` | `ealpr/extensions.py` | per process |

Only `InferenceJob` documents in MongoDB are cross-process. Everything that
governs whether work gets *done* is in-memory and therefore per-process.

---

## 4. Startup Lifecycle

### What executes at import time

`app.py:10` calls `create_app()` at module scope, **outside** the
`if __name__ == "__main__":` guard. Therefore merely importing `app` (as a WSGI
server, a test, or a management script would) executes the entire startup
sequence: MongoDB connection, blueprint registration, default-admin creation,
C2/C3 recovery, and C4 monitor start.

`config.py` has additional import-time side effects: `SAVE_DIR.mkdir()` and
`TEMP_DIR.mkdir()` (`config.py:30-31`) execute on import.

### Once per process vs. once per application instance

`create_app()` is *not* idempotent in the way C4 is. It re-registers blueprints
on a brand-new `Flask` object, re-runs `_connect_mongodb()` (which calls
`disconnect()` then `connect()`), and re-runs `_create_default_admin()`.

However, the C4 side effects **are** process-scoped and idempotent:

- `initialize_inference_jobs` (`inference_jobs.py:1167`) checks
  `_initialization_done`; with `force=False` a second call returns
  `{"skipped": True}`.
- `_ensure_worker` (`:278`) and `_ensure_monitor` (`:541`) are both idempotent
  under a lock.

**Verified statically:** calling `create_app()` twice in one process does not
create a second consumer, a second monitor, or a second recovery run. The
second app instance is a separate `Flask` object with its own `app_context`
stack, but it shares the module-level C4 state.

### Can multiple application instances duplicate workers or monitors?

**No, not within one process.** The C4 guards are module-level and
lock-protected, which was the C3.4/C4 invariant and is proven by
`test_c3_12_exactly_one_consumer_exists` and `test_c4_19`. This property is
**per-process only** — see §10.

### MongoDB initialization

`_connect_mongodb` (`ealpr/__init__.py:155-204`) attempts up to 3 connections
with exponential backoff and 5-second server-selection/connect/socket timeouts.
On total failure it sets `DB_ENABLED = False` and the app continues in a
**read-only offline mode** with a mock admin user. This is a deliberate C3.10
design decision, not a defect, but it is a production consideration: **the
application will start and serve traffic with no database**, and job submission
depends on `DB_ENABLED` being true at request time.

---

## 5. Current Server / WSGI Runtime

### What is actually used

**Werkzeug's development server.** Evidence chain, all verified:

1. `app.py:13` calls `socketio.run(...)`.
2. Installed `flask_socketio/__init__.py:641` — `if self.server.eio.async_mode ==
   'threading':` is true (see §6), so the code path is `:660`:
   `app.run(host=host, port=port, threaded=True, use_reloader=use_reloader, **kwargs)`.
3. Installed `flask/app.py:662` — `run_simple(host, port, self, **options)`.
4. Runtime probe (`c5_runtime_probe.py`, read-only, no port bound, no MongoDB
   contacted) confirmed the effective configuration:

```
host        = 0.0.0.0
port        = 5000
options_passed_to_werkzeug = {'threaded': True, 'use_reloader': False}
app.debug_at_run_time = True
eio.async_mode = threading
server() class = Server
```

### Is there a production WSGI/Socket.IO server?

**No.** Verified by exhaustive search and dependency inspection:

| Check | Result |
|---|---|
| `gunicorn` in repo or installed | **Absent** |
| `waitress` / `uvicorn` / `hypercorn` / `uWSGI` | **Absent** |
| `wsgi.py` / `gunicorn.conf.py` | **Absent** |
| `Dockerfile` / `docker-compose.yml` / `Procfile` / `Makefile` | **Absent** |
| `pyproject.toml` / `Pipfile` | **Absent** |
| `gevent` / `eventlet` | **Absent** |
| `message_queue` argument to `SocketIO()` | **Never used** |
| Production server in `requirements.txt` | **Absent** |
| Deployment documentation | **Absent** (see below) |
| `git grep -iE "gunicorn|waitress|uvicorn|hypercorn|wsgi"` in app code | **Zero hits** |

The only occurrences of those strings anywhere in the repository are inside
*previous audit reports* and one comment in `ealpr/extensions.py:14-15`.

### Deployment documentation

`README.md` contains no production, deployment, or server-instruction section.
Line 48 says only: *"Run the Application: Start the Flask server to launch the
web interface."* Line 36 also documents `OCR_MODEL_PATH` pointing at
`yolov8n_car_plate` while `config.py:10` and `.env.example` both default to
`yolo11m_car_plate` — a documentation drift, noted for completeness.

`.github/workflows/python-package.yml` is a **CI test workflow only** (flake8 +
pytest on Python 3.9/3.10/3.11). It does not build, package, or deploy anything.

---

## 6. Flask-SocketIO Runtime

| Property | Value | Evidence |
|---|---|---|
| `async_mode` | `"threading"` | `extensions.py:19` and `__init__.py:43` — declared **twice**, consistently |
| Server implementation | python-socketio `Server` | runtime probe |
| WebSocket transport | Available | `simple-websocket 1.1.0` installed; probe confirmed importable |
| Long-polling fallback | Yes | inherent to `threading` mode via engineio |
| `message_queue` | **Not configured** | `git grep` — zero hits |
| Message broker dependency | **None installed** | no redis/pika/kombu/kafka in venv |
| `cors_allowed_origins` | `"*"` | `extensions.py:19`, `__init__.py:43` |

### Where Socket.IO is actually used

| Location | Event | Thread |
|---|---|---|
| `blueprints/detection.py:279` | `socketio.emit("live_detection", …)` | live camera/video request path |
| `utils.py:562` | `socketio.emit("live_detection", …)` | video frame processing |
| `blueprints/detection.py:889` | `@socketio.on("connect")` | — |
| `blueprints/detection.py:894` | `@socketio.on("disconnect")` | — |

**Notably, the C2/C3/C4 asynchronous inference job path does not emit Socket.IO
events at all.** `inference_jobs.py` contains no `socketio` import. This is
architecturally helpful: the C4 worker is not coupled to the Socket.IO server,
so worker supervision is unaffected by Socket.IO transport problems.

### Multi-process Socket.IO

`threading` mode stores its client registry in the python-socketio `Server`
object — in-process memory. With no `message_queue`, a `live_detection` emitted
in process A reaches only clients connected to process A. Under a load balancer
with sticky sessions absent, **clients connected to process B would silently
receive no live frames**.

This is a well-known Flask-SocketIO constraint. It is recorded here as a
documented constraint, not a defect in the current single-process deployment
where it does not arise.

---

## 7. C4 Worker Interaction

### What exists today

| Component | Function | Line |
|---|---|---|
| Consumer creation | `_ensure_worker` | `inference_jobs.py:278` |
| Consumer loop | `_worker_loop` / `_worker_loop_body` | `:310` / `:321` |
| Health snapshot | `worker_health` | `:411` |
| Restart budget | `_consume_restart_budget` | `:450` |
| Restart action | `_restart_worker` | `:478` |
| Recovery retry | `_retry_recovery` | `:489` |
| Supervision tick | `_monitor_tick` | `:516` |
| Monitor loop | `_monitor_loop` | `:528` |
| Monitor creation | `_ensure_monitor` | `:541` |
| Monitor stop | `_stop_monitor` | `:558` |
| Lifecycle | `shutdown` | `:588` |
| Initialization | `initialize_inference_jobs` | `:1167` |
| Liveness predicate | `worker_is_running` | `:1240` |

### The nine requested scenarios

| # | Scenario | Observed behavior | Status |
|---|---|---|---|
| 1 | One process starts | `create_app()` → recovery → monitor started; consumer started **lazily only if recovery requeued work** | Verified (code) |
| 2 | `create_app()` called twice | Second call returns `{"skipped": True}`; no second worker, monitor, or recovery | Verified (C3/C4 tests) |
| 3 | Two processes start | Each gets its own queue, worker, monitor, `_initialization_done`. **Both run recovery. See §10 — this is the dangerous case.** | Inferred |
| 4 | One process dies | Its in-memory queue is lost. Its `queued` jobs are invisible to other processes' monitors and are only recovered at some process's next startup. | Inferred |
| 5 | One worker dies | Monitor detects `work_outstanding and not worker_running` within ≤5s and restarts via `_ensure_worker` | Verified (C4 tests) |
| 6 | One monitor dies | **Nothing restarts it.** `_monitor_loop` has a `finally` that only logs; there is no supervisor of the supervisor. C4 does not monitor itself. | Verified (code) |
| 7 | Normal shutdown (`SIGTERM`/Ctrl-C) | **No handler exists.** Python terminates; daemon threads are killed mid-operation. `shutdown()` is never invoked. | Verified (zero `signal`/`atexit` hits) |
| 8 | Force-killed (`SIGKILL`) | Identical observable outcome to #7 from the app's perspective; job is `queued` or `processing` and awaits recovery | Inferred |
| 9 | MongoDB unavailable at startup | `_connect_mongodb` fails → `DB_ENABLED=False` → recovery reports errors → `_initialization_done` stays `False` → C4 monitor retries `recover_jobs` every 5s until MongoDB answers | Verified (C4 tests) |

### Scenario 6 deserves emphasis

C4's monitor is a single unmonitored thread. If it dies, worker supervision
silently stops for the life of the process, and **nothing in the logs will
indicate that** — the last line would be the "monitor exited" info record. The
`_monitor_loop` `except` clause (`:535-536`) protects against a failing *tick*
but not against the thread ending. This is a genuine gap in the C4 design, and
it is a C4 concern rather than a C5 runtime concern — recorded here as a
**dependency**, not a C5 action item, per the preservation rules.

### Who calls `shutdown()`?

Verified: **nobody in application code.** `inference_jobs.shutdown` is referenced
only from `tests/test_inference_jobs.py`. In production it is dead code. The
`atexit` question is therefore not "should we add a hook" but "the existing hook
is never reached" (§9).

---

## 8. Debug / Reloader Analysis

### Configuration separation

Verified: `create_app()` sets **exactly one** config key,
`app.config["SECRET_KEY"]` (`__init__.py:32`). There is **no** `DEBUG`,
`TESTING`, `ENV`, or `FLASK_ENV` key anywhere in application code. `TESTING` is
set only in `tests/test_routes_readonly.py:53`.

There is therefore **no production/development separation mechanism at all**.
The single environment switch in the entire application is the literal
`debug=True` on `app.py:17`.

### The effective debug state — verified

From the installed `flask_socketio/__init__.py`:

```
:606  debug = kwargs.pop('debug', app.debug)      -> True
:608  use_reloader = kwargs.pop('use_reloader', debug)
:614  app.debug = debug                            -> True
:615  if app.debug and self.server.eio.async_mode != 'threading':  -> not taken
:660  app.run(host=host, port=port, threaded=True, use_reloader=use_reloader, ...)
```

From the installed `flask/app.py`:

```
:653  options.setdefault("use_reloader", self.debug)
:654  options.setdefault("use_debugger", self.debug)     -> True
:662  run_simple(host, port, self, **options)
```

### Consequence 1 — the interactive debugger is live on all interfaces

`app.debug = True` → `use_debugger = True` → Werkzeug's `DebuggedApplication`
with an interactive traceback console and PIN is served on `0.0.0.0:5000`.

Note the subtlety at `flask_socketio:615`: the Socket.IO-specific
`DebuggedApplication(..., evalex=True)` wrapping is skipped because
`async_mode == 'threading'`. That is **not** a mitigation — Flask's own
`app.run()` independently sets `use_debugger=True` at `flask/app.py:654`. The
debugger is active either way.

**Impact:** remote code execution capability on a network-reachable interface,
gated only by a PIN. Werkzeug's own documentation states the debugger must not
be used in production.

### Consequence 2 — the production guard is deliberately disabled

`flask_socketio:648-659` implements a safety check: if stdin is not a TTY — the
normal condition under systemd, Docker, or nohup — the library **raises
`RuntimeError`** unless `allow_unsafe_werkzeug=True`. `app.py:19` passes exactly
that flag, so the check is downgraded to a log warning.

**Impact:** the only guardrail the framework provides against accidentally
running a development server in production has been switched off, and the
server binds `0.0.0.0` rather than `127.0.0.1`.

### Consequence 3 — the reloader is one argument away from duplicating C4

`use_reloader = kwargs.pop('use_reloader', debug)` (`flask_socketio:608`) means
`use_reloader` **defaults to `debug`**. It is currently `False` only because
`app.py:18` passes it explicitly.

If that argument were removed — a very natural "cleanup", since it looks
redundant next to `debug=True` — then every source-file change would start
Werkzeug's reloader, which runs the application in a **parent process and a
fresh child process**. The child would re-import `app.py`, re-execute
`create_app()`, and get its own `_queue`, `_queued_ids`, `_worker_thread`,
`_monitor_thread` and `_initialization_done`.

The parent would also have run `create_app()`, so the parent would hold a
monitor and possibly a consumer too. Result: two consumers, two monitors, two
recovery runs, and two independent in-memory queues in two processes — which
§10 shows is the most dangerous configuration available to this system.

I attempted to demonstrate this counterfactually at runtime; the probe **hung
because the reloader actually spawned a child process**, which is itself
confirming evidence. It was terminated. The static evidence at
`flask_socketio:608` is sufficient and is what the report relies on. **No
reloader was left running.**

**This is the single most dangerous latent regression in the repository**: a
one-token edit to `app.py` silently converts the system from safe to unsafe, and
nothing in the code, tests, or documentation would flag it.

---

## 9. Shutdown / Process Lifecycle

Verified by exhaustive search across `ealpr/`, `tests/`, `config.py`, `app.py`
and `models.py` for `atexit`, `signal.`, `SIGTERM`, `SIGINT`,
`teardown_appcontext`, `teardown_request`, `weakref.finalize`:

**Zero matches. There is no shutdown handling of any kind in the repository.**

| Capability | Present? |
|---|---|
| `atexit` registration | No |
| `SIGTERM` handler | No |
| `SIGINT` handler | No |
| `teardown_appcontext` | No |
| `teardown_request` | No |
| Consumer stop on exit | No — only `shutdown()`, which is never called |
| Monitor stop on exit | No — only `_stop_monitor()` via `shutdown()` |
| Queue draining | No |
| In-flight job completion on exit | No |
| C4 startup guard reset on exit | No |

### What is actually guaranteed

Both C4 threads are `daemon=True` (`inference_jobs.py:299`, `:551`). On process
exit Python does **not** run daemon threads to completion; it simply stops the
process. So:

- An in-flight `run_job` is terminated at an arbitrary point. The job remains
  `processing` in MongoDB. The staged file remains in `TEMP_DIR`.
- Nothing flushes, nothing is logged, nothing is finalised.
- The next process's C4.1 recovery will fail that job as `orphaned_processing`
  and reclaim the staged file — **this is exactly the scenario C4.1 was built
  for, and it works.** That is the one real shutdown guarantee, and it is
  provided by *startup* recovery, not by shutdown handling.

### Practical consequence

Under systemd, Docker, or any supervisor that sends `SIGTERM`, the application
performs no cleanup and no logging. C3 and C4 are designed to tolerate this, so
this is a **hygiene and operability** gap rather than a correctness gap. It does,
however, mean the operator gets no signal that work was in flight.

---

## 10. Multi-Process Analysis

Per the audit instructions, **no multi-process deployment was executed.** This
section is static analysis. It is not a claim that multi-process works.

### 10.1 Would each process get its own queue?

**Yes — verified statically.** `_queue`, `_queued_ids`, `_all_workers`,
`_monitor_thread`, `_restart_times` and `_initialization_done` are all
module-level objects in `ealpr/inference_jobs.py` (`:258`, `:267-272`). Each
process has its own copy. A job id enqueued in process A is invisible to
process B's `worker_health()`, because `work_outstanding` is computed from
`_queue.qsize()` and `len(_queued_ids)` (`:427-440`) — both process-local.

### 10.2 Can MongoDB atomic claiming prevent duplicate inference?

**Yes — verified.** `_claim` (`:714-736`) is a single `findAndModify` filtered on
`job_id` **and** `status="queued"`. `job_id` is `unique=True` in
`models.py`, so a unique index backs it. Only one caller can win the transition
to `processing`; a second caller gets `None` and `run_job` logs
`"Ignoring job … already in state …"` and returns without inferring (`:746-753`).

**This holds across processes**, because the arbitration lives in MongoDB
rather than in process memory. **No duplicate inference would occur even with
N processes.**

### 10.3 Finding 6 — one process can fail a job another process is running

This is the most serious multi-process defect, and it is specific to C4.1.

C4.1's orphan branch (`:1070-1076`) is:

```python
if orphan_processing and not worker_is_running():
    counts["orphaned_processing"] = _fail_stale(
        lambda: InferenceJob.objects(status=InferenceJob.STATUS_PROCESSING),
        ERROR_STALE_PROCESSING, now, counts, "orphaned_processing")
```

Two properties combine badly:

1. **The guard is process-local.** `worker_is_running()` (`:1240`) inspects this
   process's `_all_workers`. In a freshly started process B, no consumer exists
   yet, so `worker_is_running()` is `False`.
2. **The query has no age filter.** Compare the age-based `stale_processing`
   branch immediately below it (`:1081-1093`), which requires
   `started_at < processing_cutoff` (600s). The orphan branch matches **any**
   `processing` job regardless of age — by design, for the single-process case
   where "no local worker" implies "no worker anywhere".

`_fail_stale` (`:887-935`) then performs a conditional bulk `update` filtered on
`status=processing`, sets `status=failed` plus `error_code`, and calls
`recover_staged_artifact` (`:933-934`) which **deletes the staged input file**
(`:875`).

**Inferred sequence with two processes:**

1. Process A claims job J (`processing`) and spends ~2.4s in inference.
2. Process B starts in that window and runs startup recovery.
3. B sees J as `processing`, concludes it is orphaned, **fails it and deletes
   its staged file**.
4. A finishes inference, calls `_persist_detection` (a real `DetectionResult` is
   written), then `_complete` → `_mark` → `job.save()` (`:630-635`).
5. `_mark` performs an **unconditional save with no status re-check**, so the
   job is flipped back to `completed` — but `error_code` and `error_message`
   set by B are **not cleared** (`:918-923` set them; `:805-813` does not
   unset them).

**Net outcome:** a `completed` job carrying a `stale_processing` error code, a
detection persisted for a job that was declared failed mid-flight, a staged
input file deleted from under a running inference, and a spurious recovery log
line. With a different interleaving the job can instead end `failed` **with a
detection present** — the exact orphaned-detection state C3/C4 were written to
reason about.

This is a **C4 semantic** issue surfaced by a runtime condition. Per the
preservation rules it is recorded here as a dependency/risk and is **not**
proposed for C5 implementation.

### 10.4 Finding 7 — `queued` jobs strand when their process dies

If process A enqueues job J and dies before its worker claims it, J remains
`queued` in MongoDB. Process B's monitor will **never** restart a worker for it,
because `work_outstanding` is process-local (§10.1) and B's queue is empty. The
job is resolved only when some process next starts and runs recovery. Recovery
is guaranteed on *startup*, not on *failure*.

This is a genuine consequence of the C4 design being per-process. It does not
occur in the current single-process deployment, where the process dying ends the
job's world and its successor recovers everything.

### 10.5 Other multi-process consequences

| Consequence | Detail |
|---|---|
| Independent recovery runs | Every process runs `recover_jobs()` at startup. Idempotent, so the result converges, but each logs its own counts. |
| Duplicate work dispatch | If B's recovery requeues a job A's recovery also requeued, both workers attempt it. `_claim` resolves it; the loser logs "Ignoring job". No duplicate inference. |
| Inconsistent worker health | `worker_health()` describes only the calling process. It is a per-process fact and is labelled as such in its docstring. Correct as written, misleading if read as global. |
| Inconsistent restart budgets | `_restarts_total` and the sliding window are per process. With N processes the effective global restart allowance is N× the configured value. |
| No cross-process Socket.IO | See §6. |
| Model memory | YOLO/OCR weights are per process, so N processes cost N× the model memory. |

### 10.6 Conclusion on process model

**The architecture should explicitly and permanently remain single-process.**

This is not a limitation to be engineered away in C5 — it is a deliberate
dependency. C4.1's orphan rule (§10.3) is *only* correct in a single-process
deployment, and multi-process makes it actively wrong. Converting to multiple
workers would require a global liveness signal (a heartbeat in MongoDB, a
lease on the job document) before the orphan rule could be trusted.

That is a design change to C2/C3/C4 semantics, which this block must not propose.
It is documented as a **deployment constraint** (§14) instead.

---

## 11. Dependency / Deployment Audit

### Declared vs. actually required

`requirements.txt` (20 lines) declares: opencv-python, numpy, Pillow,
pytesseract, ultralytics, flask, flask-sqlalchemy, flask-login, python-dotenv,
pandas, matplotlib, scikit-learn, python-dateutil, mongoengine, dnspython,
pymongo, openpyxl (pinned), python-bidi, arabic-reshaper, flask-socketio.

**Missing from `requirements.txt`: a WSGI server, and any constraint on
`flask-socketio`.** `flask-socketio` is unpinned (`flask-socketio` with no
version), as is almost everything else — the application has no upper bounds on
any core dependency. For a production runtime that is a real reproducibility
gap.

**Note on the venv:** the local `.venv` also contains `Flask-Bcrypt`,
`Flask-Limiter`, `Flask-WTF` and `Flask-PyMongo`, which are **not declared**.
Verified: **no application module imports any of them** (`git grep` for
`flask_bcrypt|flask_limiter|flask_wtf|flask_pymongo|Limiter` returns zero
hits). So `requirements.txt` is currently *adequate for imports*; the venv
simply carries unused extras.

Worth stating separately: **`Flask-Limiter` is installed but unused — the
application has no rate limiting of any kind.** For an internet-facing service
with a camera feed and file uploads, that is a production-hardening gap
belonging to a later block, not C5.

### Installed runtime versions (verified via `importlib.metadata`)

| Package | Version |
|---|---|
| Flask | 3.1.0 |
| Werkzeug | 3.1.3 |
| Flask-SocketIO | 5.5.1 |
| python-socketio | 5.13.0 |
| python-engineio | 4.12.1 |
| simple-websocket | 1.1.0 |

No gevent, no eventlet, no gunicorn, no waitress, no redis.

### Configuration findings

| Finding | Evidence | Impact |
|---|---|---|
| `SECRET_KEY` falls back to the literal `'ealpr-dev-secret'` | `config.py:57` | A deployment that omits `.env` silently runs on a public, well-known session-signing key. No validation, no startup refusal, no length check. |
| `DEFAULT_ADMIN_PASSWORD` falls back to `'admin123'` and **is not set in `.env`** | `config.py:58`; `.env` defines only `MONGODB_URI`, `MONGODB_DB_NAME`, `SECRET_KEY` | The admin account is created with a published default password. `_create_default_admin` (`__init__.py:226-227`) only *prints* a warning. |
| `.env.example` documents `retryWrites=true&w=majority`, but `create_app` forces both off | `.env.example` vs `__init__.py:184-185` (`retryReads=False`, `retryWrites=False`) | Effective durability is weaker than the documented template implies. A reader trusting `.env.example` would be misled. The explicitness is good (it is a deliberate C3.10 choice); the **documentation conflict** is the defect. |
| No `DEBUG`/`TESTING`/`ENV` config | `__init__.py:32` only | No production/development separation. |
| Import-time filesystem side effects | `config.py:30-31` | `mkdir` on import; fails if the filesystem is read-only. |
| `README.md:36` documents a different OCR model path than `config.py:10` | — | Documentation drift. |

### Deployment artifacts

**None exist.** No Dockerfile, docker-compose, Procfile, systemd unit, Makefile,
`wsgi.py`, or deployment documentation. The sole documented run command is
`python app.py`.

---

## 12. Verified vs Inferred vs Unverified Findings

### VERIFIED — executed or read directly in this audit

| # | Finding | How verified |
|---|---|---|
| V1 | Runtime is Werkzeug dev server, `threaded=True`, host `0.0.0.0`, port `5000` | Read `app.py:13-19`; read installed `flask_socketio:606-661`; runtime probe |
| V2 | `app.debug is True` at run time | Runtime probe (`app.debug_at_run_time = True`) |
| V3 | `use_debugger=True` follows from `app.debug=True` | Read installed `flask/app.py:653-654` |
| V4 | `use_reloader` is currently `False` | Runtime probe: `options_passed_to_werkzeug = {'threaded': True, 'use_reloader': False}` |
| V5 | `use_reloader` **defaults to `debug`** | Read installed `flask_socketio:608` |
| V6 | `allow_unsafe_werkzeug=True` bypasses the library's production guard | Read installed `flask_socketio:640-659`; `app.py:19` |
| V7 | `async_mode == "threading"` | `extensions.py:19`, `__init__.py:43`, runtime probe |
| V8 | No gunicorn/waitress/uvicorn/hypercorn/gevent/eventlet anywhere | `git grep`, file-existence checks, `pip list` |
| V9 | No Dockerfile/Procfile/wsgi.py/pyproject.toml/Pipfile/Makefile | `Test-Path` on each |
| V10 | No `atexit`, `signal.`, `SIGTERM`, `SIGINT`, `teardown_appcontext` in the app | Exhaustive `Select-String` over all app `.py` files — zero hits |
| V11 | `shutdown()` is never called by application code | `git grep` — only test references |
| V12 | `app.config` receives only `SECRET_KEY`; no `DEBUG`/`TESTING`/`ENV` | Read `__init__.py:32`; `git grep` for `app.config[` |
| V13 | `SECRET_KEY` set in `.env`; `DEFAULT_ADMIN_PASSWORD` is **not** | `.env` key enumeration (values not printed) |
| V14 | `_claim` is a single conditional `findAndModify`; `job_id` is unique | Read `inference_jobs.py:714-736`, `models.py` `InferenceJob` |
| V15 | C4 state is module-level and per-process | Read `inference_jobs.py:258`, `:267-272`, `:427-440` |
| V16 | The orphan branch has **no age filter**; the age branch does | Read `inference_jobs.py:1070-1076` vs `:1081-1093` |
| V17 | `_mark` saves unconditionally with no status re-check | Read `inference_jobs.py:630-635` |
| V18 | `_fail_stale` deletes the staged file of every job it fails | Read `inference_jobs.py:918-934`, `:875` |
| V19 | No `message_queue`; no broker installed | `git grep`; `pip list` |
| V20 | The async inference path emits no Socket.IO events | `git grep socketio` — no hit in `inference_jobs.py` |
| V21 | No application module imports the undeclared venv packages | `git grep` — zero hits |
| V22 | No Flask-Limiter import → no rate limiting in the app | `git grep` — zero hits |
| V23 | CI installs only `requirements.txt`; no deployment job | Read `.github/workflows/python-package.yml` |
| V24 | Working tree unmodified by this audit | `git status` clean before and after |

### INFERRED — statically derived from code, not executed

| # | Finding | Basis |
|---|---|---|
| I1 | Two processes starting can cause one to fail the other's in-flight job (§10.3) | V14+V15+V16+V17+V18 composed |
| I2 | A `queued` job whose process dies strands until some process restarts (§10.4) | V15 + `worker_health` inputs |
| I3 | A dying monitor is never restarted (scenario 6) | `_monitor_loop` `:528-538` has no supervisor |
| I4 | Force-kill is observationally identical to normal shutdown (no handlers) | V10 |
| I5 | Removing `use_reloader=False` from `app.py` would enable the reloader and duplicate all C4 state | V5 + documented Werkzeug reloader semantics |
| I6 | The Werkzeug debugger is network-reachable on `0.0.0.0`, PIN-gated | V1+V2+V3 |
| I7 | Global restart allowance becomes N× configured with N processes | V15 |

### UNVERIFIED — cannot be determined without running something unsafe

| # | Question | Why unverified |
|---|---|---|
| U1 | Does the app actually start and serve under a real production server? | No server is installed; installing one is out of scope. |
| U2 | Does the Werkzeug debugger actually accept PIN guesses in this deployment? | Requires a running server; not started. |
| U3 | Does multi-process actually produce the §10.3 corruption? | Would require a 2-process deployment; explicitly excluded by the audit brief. |
| U4 | Actual sustained throughput/load behaviour under production traffic | No load was generated. |
| U5 | Behaviour under systemd/Docker signal delivery | No such configuration exists to test. |
| U6 | Whether the current `.env` MongoDB URI's Atlas tier tolerates the C3 5-second socket timeouts | Would require production load; not exercised. |
| U7 | Whether the PIN is guessable from this host's public files | Out of scope and security-adjacent. |

---

## 13. Risks

Ordered by severity. All are process-layer risks; **none originate in C2/C3/C4
logic**.

| # | Risk | Severity | Exploitability / Likelihood | Status |
|---|---|---|---|---|
| R1 | Werkzeug interactive debugger served on `0.0.0.0:5000` — remote code execution, PIN-gated | **Critical** | Reachable by anyone who can reach the port; PIN is the only barrier | Verified present |
| R2 | `allow_unsafe_werkzeug=True` disables the framework's only production guard | **High** | Certain once deployed | Verified present |
| R3 | No production server: dev server has no request limits, no keep-alive tuning, no worker pool, and is explicitly unsupported by its authors | **High** | Certain | Verified present |
| R4 | One-token edit (`use_reloader=False`) silently creates a dual-process, unsafe deployment | **High** | Requires a plausible "cleanup" commit | Verified latent |
| R5 | No shutdown handling: in-flight work is killed with no log, no drain, no signal to the operator | **High** | Certain on every restart | Verified present |
| R6 | Multi-process corrupts job state via the C4.1 orphan rule (§10.3) | **High** | Certain if anyone deploys multi-process | Inferred |
| R7 | Multi-process strands `queued` jobs (§10.4) | **High** | Certain if multi-process | Inferred |
| R8 | Default admin password `admin123`, warning only | **High** | Trivial to guess | Verified present |
| R9 | `SECRET_KEY` silently falls back to a published literal | **High** | Certain if `.env` is missing | Verified present |
| R10 | No rate limiting anywhere; camera feed and upload endpoints exposed | Medium | Depends on network exposure | Verified absent |
| R11 | Live Socket.IO broadcasts silently break under multi-process | Medium | Certain if multi-process | Verified by configuration |
| R12 | No documented deployment procedure at all | Medium | Certain | Verified present |
| R13 | Unpinned core dependencies; no reproducible production build | Medium | Certain | Verified present |
| R14 | `DB_ENABLED=False` fallback lets the app serve with no database | Medium | Only during a MongoDB outage | Verified by design (C3.10) |
| R15 | `.env.example` durability guidance contradicts actual settings | Low | Misleading documentation | Verified present |
| R16 | A dying C4 monitor disables supervision silently (§7 scenario 6) | Medium | Requires monitor-thread death | Inferred; **C4 concern** |
| R17 | CI is not a real gate: no coverage requirement, and `python-package.yml` uses Python 3.9 while the venv is 3.10 | Low | — | Verified |

---

## 14. Recommended C5 Scope

Derived strictly from the findings above. The guiding constraint: **fix the
process layer, touch nothing in the job layer.**

### In scope for C5

| Item | Addresses | Rationale |
|---|---|---|
| **S1. Environment-driven runtime mode** — introduce an explicit `EALPR_ENV`/`EALPR_DEBUG` style switch with a real default, and make the app refuse to start with `debug=True` unless explicitly in development | R1, R2, R12 | The single highest-value change. Removes the accidental-debugger class of risk entirely and replaces a hardcoded literal with intent. |
| **S2. Production server entry point** — add a documented, tested production run command using a server that can actually be justified for this application | R3, R12, R13 | Currently no production runtime exists at all. **Requires a decision from the application owner**: no WSGI server is installed and choosing one is a deployment decision, not an audit decision. Must be paired with §6 (`message_queue`) if more than one process is ever used. |
| **S3. Single-process enforcement/documentation** — state the single-process constraint in code and documentation, and make any multi-process configuration fail loudly rather than silently | R4, R6, R7, R11 | §10.6 shows multi-process is not merely unsupported, it is *incorrect*. Silent failure is the real hazard. Cheapest high-value fix. |
| **S4. Wire `atexit` to the existing `shutdown()`** — one registration, no new shutdown logic | R5, V11 | The function already exists and is already correct; it is simply never called. This is a ~3-line change with disproportionate benefit. |
| **S5. Secret and credential startup validation** — refuse to start in production with the default `SECRET_KEY` or default admin password; document the requirement | R8, R9 | Closes two "silent insecure default" paths without changing any secret values. |
| **S6. Startup/runtime logging discipline** — replace `print()` in the startup path with the application logger, so production start-up is observable | R5, R12 | `_connect_mongodb` and `_create_default_admin` currently `print()`. Low risk, directly improves operability. |
| **S7. Dependency pinning and deployment documentation** — pin the core runtime, add a real deployment section to `README.md` | R12, R13, R15 | Includes correcting the `.env.example` retry-write guidance. |

### Explicitly NOT in scope for C5

Deferred deliberately, with reasons — see §15.

### Ordering constraint

**S1 and S3 should land before S2.** Turning off `debug` while still on the
development server is a safe intermediate state; introducing a production server
while `debug=True` is not, because the debugger is enabled through
`app.debug`, which a WSGI server also honours.

### Suggested verification for C5

Tests should assert: the app refuses to start with `debug=True` in production
mode; the reloader cannot be enabled implicitly; `atexit` invokes
`shutdown()`; a multi-worker server configuration raises a clear error;
secrets are validated at startup. These are all testable without MongoDB
mutation by importing `app.py` in a subprocess with environment overrides.

---

## 15. Explicit Out-of-Scope Items

Recorded so that C5 is not silently expanded. Each of these was considered and
deliberately excluded.

| Item | Why excluded |
|---|---|
| **C4.1 orphan rule** (`recover_jobs(orphan_processing=…)`) | A C4 semantic. §10.3 shows it is *correct only in a single process* — which is precisely why S3 (enforce single process) is the correct response, rather than changing C4. |
| **Atomic `_claim`**, `queued → processing` semantics, `processing`-job requeue behaviour, `_persist_detection`, `_complete`, inference pipeline, YOLO models, OCR, camera/video, UI, authentication, repositories, MongoDB schema, image storage, C2/C3 job semantics | **C4 Preservation Rules.** None of these require a production-runtime change. The `_mark` unconditional save noted in V17 is a C4 concern and is reported, not proposed for change. |
| **Monitor self-supervision** (§7 scenario 6, R16) | A C4 reliability gap, not a runtime one. Reported as a dependency for a future C4 revision. |
| **Adding a message broker** (Redis) | The brief forbids it, and §10.6 concludes multi-process should not be pursued. Adding a broker would imply a multi-process target that does not exist. |
| **Rate limiting** (R10) | Real gap, but a feature block, not runtime hardening. `Flask-Limiter` is already installed. |
| **Choosing the specific WSGI server** (S2) | Requires an owner decision. The audit's job is to establish that a decision is required, not to make it. |
| **CI/CD quality** (R17) | Separate concern; no deployment pipeline exists to improve. |
| **MongoDB durability settings** (`retryWrites`) | A deliberate C3.10 choice; only the `.env.example` documentation is wrong (S7). |
| **Model/AI configuration drift** (`README.md:36`) | Documentation only; folded into S7 as an aside. |
| **Any dependency installation, config change, or commit** | Prohibited for this block. |

---

## 16. Verification Performed

All checks were read-only. **No MongoDB write, no inference job, no detection,
no visitor, no configuration change, and no modification of any repository file
occurred.**

| Check | Type | Result |
|---|---|---|
| `git log --oneline -1`, `git status --short` | Git | Baseline `c1814880`; tree clean before and after |
| Full read of `app.py` (20 lines) | Read | — |
| Full read of `ealpr/__init__.py` (229 lines) | Read | — |
| Full read of `config.py`, `ealpr/extensions.py`, `ealpr/runtime_config.py` | Read | — |
| Full read of `requirements.txt`, `.env.example` | Read | — |
| Targeted reads of `inference_jobs.py` at 255-374, 404-534, 534-608, 630-643, 714-753, 805-834, 851-935, 1055-1094, 1167-1238 | Read | — |
| Targeted read of `models.py::InferenceJob` | Read | — |
| Read of installed `flask_socketio/__init__.py:604-673` | Read | — |
| Read of installed `flask/app.py:546-662` | Read | — |
| `pip list` filter for servers/brokers/socketio | Dependency | — |
| `importlib.metadata` version query | Dependency | — |
| Existence check: Dockerfile, Procfile, wsgi.py, pyproject.toml, Pipfile, Makefile, docker-compose.yml, gunicorn.conf.py, `.github/workflows` | Filesystem | — |
| `git grep -iE "gunicorn|waitress|uvicorn|hypercorn|wsgi\|gevent\|eventlet\|redis"` | Static | No app-code hits |
| `git grep -E "socketio\.(emit\|on\|init_app\|run)"` | Static | 4 app sites enumerated |
| `git grep -E "atexit\|signal\.\|SIGTERM\|SIGINT\|teardown_appcontext"` | Static | **Zero hits** |
| `git grep -E "app\.config\[|DEBUG\|TESTING\|ENV\b\|FLASK_ENV"` | Static | 1 app config key |
| `git grep -E "flask_bcrypt\|flask_limiter\|flask_wtf\|flask_pymongo\|Limiter"` | Static | Zero hits |
| `.env` key enumeration (names + lengths only, **no values printed**) | Config | 3 keys; `DEFAULT_ADMIN_PASSWORD` absent |
| `c5_runtime_probe.py` — real `socketio.run()` call with `flask.Flask.run` monkeypatched so **no port was bound** and no application module imported | Runtime | Effective config captured; counterfactual reloader attempt **hung, was terminated**, no reloader left running |
| Full read of `.github/workflows/python-package.yml` | Read | CI-only |

**Explicitly not run**, because they would mutate live data or exceed audit
scope: the test suite, any `create_app()` invocation against the live
MongoDB, Block A/Block B regressions, integrity scripts, and any
multi-process deployment.

---

## 17. Final Audit Verdict

The C2 → C3 → C4 progression has produced a genuinely well-engineered **job
layer**: atomic claiming, durable recovery, no duplicate inference, structured
logging discipline, and a supervision thread that is cheap, idempotent and
honest about its own limits. Nothing in this audit asks to change any of it, and
nothing in it is at fault.

The gap is what surrounds that layer. C4 optimised the behaviour *inside* one
process and never established the process itself as a production-grade unit.
The application currently runs a development server with an interactive debugger
on all network interfaces, with the framework's own production guard explicitly
disabled, with no shutdown handling of any kind, with insecure default
credentials, and with no documented statement of the single-process constraint
that its own recovery logic depends on.

The most urgent items are small and cheap:

- **R1/R2** — make `debug=True` require explicit development intent. This is a
  one-line policy change that removes a critical remote-execution exposure.
- **R4** — enforce the single-process constraint, because the current
  `use_reloader=False` argument is the *only* thing standing between a routine
  cleanup commit and a dual-process deployment that corrupts job state.
- **R5** — call the `shutdown()` that already exists.

The most consequential item, **S2**, is a decision the audit cannot make: which
production server this application should run on. It needs the application
owner's input, and it should be sequenced after S1 and S3.

**C5 implementation has NOT been started. No commit was created. No application
code was modified.**

---

### C4 Preservation Confirmation

This audit recommends **no change** to any of the following, and each was
examined and deliberately left alone:

- atomic `_claim` — correct, and the only reason multi-process would not
  duplicate inference
- `queued → processing` semantics
- `processing`-job requeue behaviour (which is: never requeue)
- `_persist_detection`
- `_complete` / `_mark`
- the inference pipeline, YOLO models, OCR
- camera / video / UI / authentication
- repositories, MongoDB schema, image storage
- C2 and C3 job semantics, and the entire C4 worker/monitor mechanism

Where a production-runtime issue touches these — specifically §10.3, where C4.1's
orphan rule is only valid in a single process — it is documented as a
**dependency and a risk**, and the recommended response is a runtime
constraint (§14 S3), not a change to the mechanism.

# PHASE 2 — BLOCK C5, ITEM S2

# Production Server Compatibility Audit

> **READ-ONLY AUDIT.**
>
> **S2 implementation was NOT performed.**
>
> No package was installed. `requirements.txt` was not modified. No application
> code, `config.py`, `app.py`, `ealpr/inference_jobs.py`, Socket.IO code or test
> was modified. No WSGI entry point, Docker file, Procfile or systemd unit was
> added. Worker count was not changed. C4 was not changed. Nothing was committed.
>
> The only file this audit created is this report.

| | |
|---|---|
| Repository | `D:\EALPR SYSTEM` |
| Baseline commit | `dcd2e601` — *phase2: harden production secrets logging and docs* |
| Audit type | Read-only compatibility analysis |
| Data integrity | Unchanged — digest `9200c7588833bb5456221036de1d6913` |
| Outcome | Recommendation + implementation plan for a future block |

---

## 1. Executive summary

The application currently runs on **Werkzeug's development server**, reached
through `socketio.run(..., allow_unsafe_werkzeug=True)` in `app.py`. C5 S1/S3/
S4/S5/S6/S7 have already removed the debugger-by-default, multi-process and
insecure-default-credential exposures, but the *server itself* is still a
development server with the framework's own production guard explicitly
disabled. That is audit finding R3 and it is what S2 exists to close.

The decisive architectural fact, established by reading the installed sources
rather than by convention, is this:

> **In `async_mode="threading"`, Flask-SocketIO's WebSocket transport is
> provided by `simple_websocket`, and `simple_websocket` can only obtain a
> socket from four WSGI server environments: Werkzeug, Gunicorn, Eventlet and
> Gevent. Everything else raises `RuntimeError('Cannot obtain socket from WSGI
> environment.')`.**

Verified at `simple_websocket/ws.py:264-290` and stated in its own docstring at
`ws.py:309-311`.

That single fact separates the candidates more sharply than any popularity or
performance argument:

- **Waitress** satisfies every hard constraint in the brief — one process, one
  worker, no reloader, no broker, threaded, cross-platform — but it is *not* one
  of the four, so the Socket.IO **WebSocket transport** fails and every client
  falls back to **long-polling**. Long-polling still delivers every
  `live_detection` event; it is a transport change, not a feature loss.
- **Gunicorn** is one of the four, so WebSocket works — but it needs a master
  plus a worker (two OS processes), it cannot run on Windows (the platform this
  project is actually developed and verified on), and its `--workers` flag is
  completely invisible to `config.assert_single_process()`, so C5's S3 guard
  would not catch a multi-worker misconfiguration.
- **The Gunicorn/gevent and Gunicorn/eventlet recipe** that Flask-SocketIO's
  own documentation traditionally suggests is **rejected outright**: it
  requires changing `async_mode`, and `ealpr/extensions.py:13-16` records that
  this project already tried that direction and that it "causes the whole app to
  freeze when any background thread tries to do I/O."

There is also a concurrency trap that applies to *both* real candidates and is
the most operationally dangerous finding in this audit:

> **Every idle long-polling Socket.IO client pins one server thread for up to
> `ping_interval + ping_timeout` = 25 + 20 = 45 seconds**
> (`engineio/socket.py:12-17`, defaults confirmed on the installed
> `engineio.Server`).

Werkzeug's `threaded=True` creates an unbounded thread per request, which is
why the current setup tolerates many clients. Any bounded-pool server —
Waitress's default `threads=4`, Gunicorn `gthread`'s default `--threads=1` —
would be **exhausted by four (or one) idle Socket.IO clients**, at which point
the entire HTTP application, including the login page, stops responding. Moving
to a production server without raising the thread count is a availability
regression, not an upgrade.

**Recommendation: Waitress, with an explicitly configured thread pool and a
documented long-polling expectation.**
**Second best: Gunicorn with `--workers 1 --worker-class gthread`, Linux only.**

---

## 2. Current architecture

### 2.1 Entry path — verified from source

```
python app.py                                <-- the only documented run command
  │
  ├─ app.py:22      app = create_app()       (module scope, runs on ANY import)
  │    └─ ealpr/__init__.py:67 create_app()
  │         ├─ :74   _ensure_startup_logging()          (S6)
  │         ├─ :81   resolve_runtime_mode()             (S1)
  │         ├─ :88   assert_single_process()            (S3)
  │         ├─ :94   assert_production_secrets(...)     (S5)
  │         ├─ :112  app.debug = <validated>
  │         ├─ :131  socketio.init_app(async_mode="threading")
  │         │        └─ flask_socketio/__init__.py:255  app.wsgi_app = _SocketIOMiddleware
  │         ├─ :134  _connect_mongodb()
  │         ├─ :224  _create_default_admin()
  │         └─ :237  register_atexit_shutdown() + initialize_inference_jobs(app)   (C3/C4 + S4)
  │
  └─ app.py:25      socketio.run(...)        ONLY under if __name__ == "__main__"
                     host=0.0.0.0, port=5000
                     debug = app.config["EALPR_DEBUG"]
                     use_reloader=False
                     allow_unsafe_werkzeug=True
                      └─ flask_socketio/__init__.py:660
                           app.run(threaded=True, use_reloader=False)
                            └─ werkzeug.serving.run_simple   <-- DEVELOPMENT SERVER
```

Two properties of this diagram drive the whole audit:

1. **`app = create_app()` is at module scope.** Any WSGI server that imports
   module `app` executes the complete startup sequence — S1/S3/S5 checks,
   Mongo connection, admin bootstrap, C3 recovery, C4 monitor and consumer
   registration, S4 `atexit` hook — exactly once per import, per process.
2. **The `socketio.run()` block is behind `if __name__ == "__main__":`.** A WSGI
   server imports `app` rather than running it as `__main__`, so `socketio.run`
   — and therefore `host="0.0.0.0"`, `port=5000`, `use_reloader` and
   `allow_unsafe_werkzeug` — is **never executed**. The listener, its address
   and its threading are entirely the new server's configuration.

### 2.2 Verified runtime versions

| Package | Version | Present |
|---|---|---|
| Flask | 3.1.0 | yes |
| Werkzeug | 3.1.3 | yes |
| Flask-SocketIO | 5.5.1 | yes |
| python-socketio | 5.13.0 | yes |
| python-engineio | 4.12.1 | yes |
| simple-websocket | 1.1.0 | yes |
| gunicorn | — | **ABSENT** |
| waitress | — | **ABSENT** |
| gevent / eventlet / gevent-websocket | — | **ABSENT** |
| redis / celery | — | **ABSENT** |

No candidate server is installed. Nothing was installed by this audit.

### 2.3 Process-local state that a second process would duplicate

| Object | Location | Scope |
|---|---|---|
| `_queue`, `_queued_ids` | `inference_jobs.py:258-272` area | per process |
| `_all_workers`, `_worker_thread` | `inference_jobs.py:297-300` | per process |
| `_monitor_thread`, restart budget | `inference_jobs.py:549-552` | per process |
| `_initialization_done` | `inference_jobs.py` | per process |
| `INFERENCE_LOCK` | `ai_models.py:15` | per process |
| YOLO / OCR weights | `ealpr/ai_models.py` | per process |
| python-socketio client registry | in-process `Server` object | per process |
| `DB_ENABLED` | `ealpr/extensions.py:20` | per process |

Only `InferenceJob` documents in MongoDB are cross-process. Every mechanism
that decides whether work gets *done* is in-memory.

### 2.4 Where Socket.IO actually carries data — verified

| Site | Thread | Direction |
|---|---|---|
| `ealpr/utils.py:562` `socketio.emit("live_detection", …)` | `_feed_worker`, a `threading.Thread` started at `utils.py:510` | server → client |
| `ealpr/blueprints/detection.py:279` `socketio.emit("live_detection", …)` | `_run_video_job`, started as `target=` at `detection.py:361` ("Video processing runs in a background thread so it never blocks the Flask request thread") | server → client |
| `detection.py:889/894` `@socketio.on("connect"/"disconnect")` | request thread | client → server |

Client side: `templates/base.html:179` loads `socket.io.js 4.0.1`;
`base.html:236` is `const socket = io();` — **default options, no `transports`
restriction**, so the client negotiates polling first and then probes for a
WebSocket upgrade. `inference_jobs.py` contains no `socketio` import, so C4
neither emits nor consumes Socket.IO events.

---

## 3. Hard constraints

Treated as non-negotiable throughout this audit, per the brief:

| # | Constraint | Enforced by today |
|---|---|---|
| H1 | exactly **one process** | `config.assert_single_process()` (`config.py:168`) |
| H2 | exactly **one worker** | `assert_single_process()` rejects `EALPR_WORKERS != 1` |
| H3 | **reloader OFF** | `assert_single_process()` rejects truthy `EALPR_RELOADER`; `app.py:32` passes `use_reloader=False` |
| H4 | one in-process inference queue | `inference_jobs.py` module state |
| H5 | one inference consumer | `_ensure_worker` + `_initialization_done` |
| H6 | one C4 supervision monitor | `_ensure_monitor` |
| H7 | process-local worker supervision | C4 design |
| H8 | Flask-SocketIO `async_mode="threading"` | `extensions.py:17`, `__init__.py:131` — declared twice |
| H9 | no message broker / Redis / Celery / RQ / Kafka / RabbitMQ | none installed, no `message_queue` argument anywhere |

**No candidate is recommended unless it satisfies all nine.** Where a candidate
deviates — Gunicorn's master process, Gunicorn's `--workers` blind spot — that
deviation is called out explicitly rather than waved through, and it is the
reason Gunicorn is not first.

---

## 4. Candidate comparison table

| Criterion | **Waitress** | **Gunicorn `gthread` `w=1`** | **Gunicorn `sync` `w=1`** | **Gunicorn gevent/eventlet** | **Status quo: Werkzeug** |
|---|---|---|---|---|---|
| Exactly one OS process | ✅ 1 | ⚠️ 2 (master + worker; master runs no app code) | ⚠️ 2 | ⚠️ 2 | ✅ 1 |
| Exactly one worker | ✅ n/a — no worker concept | ✅ `--workers 1` | ✅ `--workers 1` | ✅ | ✅ |
| Runs on Windows (this environment) | ✅ | ❌ requires POSIX `fork`/signals | ❌ | ❌ | ✅ |
| `create_app()` executes once | ✅ | ✅ if `--preload` **off** | ✅ if `--preload` off | ✅ | ✅ |
| Reload available | ❌ none | ⚠️ `--reload` exists, **must stay off** | same | same | ⚠️ guarded by H3 |
| `async_mode="threading"` untouched | ✅ | ✅ | ✅ | ❌ **requires change** (H8) | ✅ |
| WebSocket transport | ❌ `RuntimeError` → falls back to polling | ✅ expected (`gunicorn.socket`) — **unverified, not installed** | ⚠️ connects, but blocks the only worker and trips `--timeout` | ✅ | ✅ |
| Long-polling transport | ✅ | ✅ | ✅ | ✅ | ✅ |
| Concurrent requests while a client is idle-polling | ✅ with `threads=` raised | ✅ with `--threads=` raised | ❌ **1 at a time** | ✅ | ✅ (`threaded=True`, unbounded) |
| Visible to `assert_single_process()` | ✅ no conflicting knob | ❌ **blind to `--workers`** | ❌ blind | ❌ blind | ✅ |
| Message queue required (H9) | ❌ no | ❌ no with 1 worker | ❌ no | ❌ no | ❌ no |
| C4 threads/atexit unaffected | ✅ | ✅ | ✅ | ⚠️ monkey-patching | ✅ |
| Production-grade | ✅ | ✅ | ❌ documented as unsuitable here | ⚠️ | ❌ dev server, R3 |
| Retires `allow_unsafe_werkzeug` (R2) | ✅ | ✅ | ✅ | ✅ | ❌ |
| Retires interactive debugger (R1) | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## 5. Gunicorn analysis

### 5.1 Process model

Gunicorn is a **prefork** server: a master process that binds the socket and
supervises, plus `--workers N` child processes that each import and serve the
application.

- With `--workers 1` there is **one application process**. The master does not
  import `app` (absent `--preload`), does not connect to MongoDB, does not own
  a queue, and does not touch C4 state.
- Counted as OS processes, however, it is **two**, which is a literal deviation
  from H1. The *rationale* of H1 — one queue, one consumer, one recovery owner,
  one Socket.IO registry — is satisfied, because none of those live in the
  master. The deviation is real but it is not one of the eight.
- **Application-process duplication risk:** none at `--workers 1` without
  `--preload`. Both of the following are genuine hazards:
  - **`--preload` must never be used.** It imports the application in the
    master *before* forking. `create_app()` would have already started the C4
    consumer and monitor threads and registered the `atexit` hook; **threads do
    not survive `fork()`**, so the child would inherit `_worker_thread` and
    `_monitor_thread` objects pointing at threads that no longer exist. C4's
    `_ensure_worker`/`_ensure_monitor` check `is_alive()` and would likely
    restart them, but `_initialization_done` would also have been inherited as
    `True`, which suppresses the recovery pass the fresh process actually
    needs. Unverified in this environment (Gunicorn not installed), but the
    fork semantics are not in doubt.
  - **`--reload` must never be used.** It re-executes the master on code
    change. During a graceful worker replacement the old and new workers can
    coexist briefly — two application processes, two consumers, two monitors,
    two recovery passes. That is exactly the C4.1 orphan-rule hazard the C5
    audit documented as finding 6/R6.
- **Reload:** Gunicorn's is `--reload` (file watching, master re-exec). It is
  not controlled by `EALPR_RELOADER`, so H3's guard does not cover it.

### 5.2 Flask-SocketIO compatibility

- `async_mode="threading"` works under Gunicorn; Flask-SocketIO's `run()` is
  not involved when a WSGI server serves the app, and `init_app` has already
  installed `_SocketIOMiddleware` on `app.wsgi_app`
  (`flask_socketio/__init__.py:253-255`), so `app` is a complete WSGI callable
  that answers `/socket.io/`. **No Socket.IO configuration change is required.**
- **WebSocket:** installed `simple_websocket/ws.py:268-271` explicitly looks
  for `'gunicorn.socket'` in the WSGI environ and sets `self.mode = 'gunicorn'`;
  `engineio/async_drivers/_websocket_wsgi.py:15-16` has a dedicated
  `if self.ws.mode == 'gunicorn'` branch. So the *transport layer* is designed
  to support Gunicorn. **This could not be verified by execution because
  Gunicorn is not installed and installing it was forbidden.**
- **Worker class is decisive.** With the default `sync` worker a Gunicorn worker
  handles one request at a time. `_websocket_wsgi.SimpleWebSocketWSGI.__call__`
  runs the entire WebSocket exchange synchronously and does not return until the
  socket closes, so **one WebSocket client would occupy the single worker
  forever** — the application would serve no HTTP at all. Worse, Gunicorn's
  default `--timeout 30` is enforced through a heartbeat file that a sync worker
  only updates between requests, so the worker would be **killed as a timeout**
  while a client is simply watching the feed. `--worker-class gthread` is
  therefore not optional, it is mandatory.
- `gevent` / `eventlet` worker classes would satisfy WebSocket but are rejected
  in §15.
- **Long-polling** works under any worker class.
- **No message queue is required** at one worker (H9 satisfied). Broadcasts
  remain correct precisely because there is one application process: the
  python-socketio client registry is in-process and every client is attached to
  that process.

### 5.3 EALPR C4 compatibility

Assuming `--workers 1 --worker-class gthread --preload off --reload off`:

| Component | Effect |
|---|---|
| inference worker | Started inside `create_app()` in the single worker process. Unaffected. |
| inference monitor | Same process, same thread. Unaffected. |
| in-memory queue | One process → one queue. Correct. |
| C4 restart supervision | Thread-level, not process-level. Unaffected. |
| startup recovery | Runs once, on worker import. Correct. |
| `shutdown()` / `atexit` | Registered in the worker process; runs when that interpreter shuts down normally. |
| SIGTERM | Delivered by the master to the worker. Whether the worker exits *normally* (so `atexit` runs) or is terminated is **unverified** — Gunicorn not installed. |
| process termination | A crashed worker is restarted by the master, which re-imports `app` → re-runs `create_app()` → re-runs C3 recovery. This is a genuine robustness **advantage** over Waitress, which has no supervisor. |

**No C4 change is required, and none is proposed.**

### 5.4 Application startup

- `create_app()` executes **once**, in the worker, when Gunicorn imports
  `app:app`.
- **Not** in a parent and child (without `--preload`).
- **Not** under reload, provided `--reload` is off.
- With `--preload`: once in the master, then inherited by the child in a broken
  state — see §5.1.

### 5.5 Runtime configuration

| Variable | Compatibility |
|---|---|
| `EALPR_ENV` | ✅ inherited by the worker; `config.py` reads it at import; S5 applies |
| `EALPR_DEBUG` | ✅ honoured; S1 still refuses it outside development |
| `EALPR_WORKERS` | ⚠️ **This is the conflict.** `assert_single_process()` reads `EALPR_WORKERS`, which Gunicorn does not read; Gunicorn reads `--workers`. Setting `EALPR_WORKERS=1` while running `gunicorn -w 4` passes the guard and violates H1. The guard provides **no protection at all** against Gunicorn-level fan-out. |
| `EALPR_RELOADER` | ⚠️ Same class of gap: Gunicorn's `--reload` is a separate knob the guard does not see. |

`assert_single_process()` cannot be extended to read a server's own flags
without knowing which server is running, and extending it is an application
change outside this audit's authority.

### 5.6 Entry point

`python app.py` **cannot** remain the production entry point: executing it runs
the `__main__` block, which starts Werkzeug. The minimum change is to serve the
already-existing WSGI callable instead, e.g.
`gunicorn --workers 1 --worker-class gthread --bind 127.0.0.1:5000 'app:app'`.
No new file is strictly required, because `app.py:22` already exposes `app`.
Gunicorn requires a POSIX platform; **it cannot run in the environment this
project currently occupies.**

### 5.7 Shutdown

- SIGINT → `KeyboardInterrupt` in the main thread → worker exits normally →
  `atexit` → `shutdown()`. Expected.
- SIGTERM → master signals worker → worker's main loop exits → normal shutdown
  → `atexit` → `shutdown()`. Expected, **unverified**.
- `--graceful-timeout` bounds the drain.
- Worker crash → master restarts → C3 recovery on the new process.

### 5.8 Failure behaviour

| Event | Behaviour |
|---|---|
| Worker crash | Master restarts it; recovery re-runs. **Best of all candidates.** |
| Master crash | Everything dies; supervisor restarts the master. |
| SIGTERM | Graceful, expected to reach `atexit` (unverified). |
| SIGINT | Graceful, reaches `atexit`. |
| Reload | Transient dual-process → C4.1 hazard. **Must stay off.** |
| Abnormal termination (`SIGKILL`) | No cleanup; C3/C4 startup recovery resolves the in-flight job. Same as today. |

---

## 6. Waitress analysis

### 6.1 Process model

- **Exactly one OS process.** No fork, no master, no worker processes.
  Waitress creates a listener thread and a fixed pool of task threads *inside*
  that one process.
- **No worker-count concept**, so there is no server-side flag that could
  violate H1/H2, and nothing for `assert_single_process()` to disagree with.
  The guard and the server are structurally compatible.
- **No reloader.** Nothing can be turned on. H3 is satisfied by construction.
- **Runs on Windows, Linux and macOS** — pure Python, no platform-specific
  syscall. It is the only production candidate that runs in the environment this
  project currently occupies.
- **Application-process duplication risk: none.**

### 6.2 Flask-SocketIO compatibility

- `async_mode="threading"` is unchanged and appropriate: Waitress is a
  thread-per-task server and does not monkey-patch anything. H8 satisfied.
- **Long-polling: fully supported.** It is plain WSGI `GET`/`POST` to
  `/socket.io/`, routed by `_SocketIOMiddleware` which `init_app` already
  installed (`flask_socketio/__init__.py:253-255`). **No Socket.IO code or
  configuration change is required.**
- **WebSocket: NOT supported.** Verified, not inferred:
  - `engineio/async_drivers/threading.py` sets
    `_async['websocket'] = SimpleWebSocketWSGI`, so `engineio.base_server`
    advertises `upgrades: ['websocket']` to every client.
  - `engineio/socket.py:142-149` then constructs `SimpleWebSocketWSGI(environ)`
    on the upgrade request.
  - `simple_websocket/ws.py:264-290` checks, in order, `'werkzeug.socket'`,
    `'gunicorn.socket'`, `'eventlet.input'`, and a gevent `SERVER_SOFTWARE`.
    Waitress supplies **none** of them, so `sock is None` and the constructor
    raises `RuntimeError('Cannot obtain socket from WSGI environment.')`.
    The package's own docstring (`ws.py:309-311`) confirms: *"Werkzeug,
    Gunicorn, Eventlet and Gevent are the only web servers that are currently
    supported."*
  - The error is raised inside the WSGI call with no surrounding `try/except`,
    so the upgrade request returns **HTTP 500**. Engine.IO's transport
    negotiation then leaves the client on long-polling — that is the designed
    fallback, and `templates/base.html:236` (`const socket = io();`) uses
    default options, so it will take it.
- **Functional consequence:** every `live_detection` event still reaches every
  connected client. What is lost is the WebSocket channel: one extra HTTP
  round-trip per packet, and an error line in the server log on every client
  connect from the failed upgrade probe.
- **No message queue is required** (H9). Broadcasts stay correct because there
  is exactly one process and every client is attached to it.

### 6.3 EALPR C4 compatibility

| Component | Effect |
|---|---|
| inference worker | Started in `create_app()` inside the one Waitress process. Unaffected. |
| inference monitor | Same. Unaffected. |
| in-memory queue | One process → correct. |
| C4 restart supervision | Thread-level. Unaffected. |
| startup recovery | Runs once at import. Correct. |
| `shutdown()` / `atexit` | Registered in the same process; runs on normal interpreter shutdown. |
| SIGTERM | No handler exists in the application (`git grep signal.` → zero hits) and none is added by Waitress's API surface as documented. Default disposition terminates the process **without** running `atexit`. |
| SIGINT | Python raises `KeyboardInterrupt`; the main thread unwinds; `atexit` runs → `shutdown()`. |

**No C4 change is required, and none is proposed.**

### 6.4 Application startup

- `create_app()` executes **exactly once**: Waitress imports `app` once, in one
  process, at startup.
- Not in a parent and child — there is no parent/child.
- Not under reload — there is no reload.
- Not twice under worker startup — there are no workers.

### 6.5 Runtime configuration

| Variable | Compatibility |
|---|---|
| `EALPR_ENV` | ✅ inherited; read by `config.py` at import; S5 applies |
| `EALPR_DEBUG` | ✅ honoured; S1 still refuses it outside development |
| `EALPR_WORKERS` | ✅ Waitress does not read it; `1` (or unset) is accepted by `assert_single_process()` and there is no competing flag |
| `EALPR_RELOADER` | ✅ Waitress has no reloader; default off accepted |

**No conflict with `assert_single_process()`. This is the cleanest fit of any
candidate.**

### 6.6 Entry point

`python app.py` **cannot** remain the production entry point, because running
it executes `socketio.run()` → Werkzeug. The production command becomes a
server invocation against the WSGI callable that already exists at
`app.py:22` — e.g. `waitress-serve --listen=… --threads=N app:app`. No new
entry-point file is strictly required. `python app.py` correctly remains the
**development** command.

### 6.7 Shutdown

| Signal / event | Behaviour |
|---|---|
| SIGINT (Ctrl-C) | `KeyboardInterrupt` → main thread exits → interpreter shutdown → `atexit` → `shutdown()` stops consumer and monitor. |
| SIGTERM | Default disposition → process terminated → **`atexit` does not run** → `shutdown()` not called → C4 threads die mid-operation → in-flight job left `processing` → resolved by C3/C4 recovery on next start. |
| Normal exit / supervisor stop | `atexit` runs. |
| Process crash | Same as SIGTERM; recovery on next start. |
| Reload | N/A. |

**This is identical to the current behaviour.** Werkzeug's
`serving.py` contains no signal handlers either (verified: only
`serving.py:820 except KeyboardInterrupt: pass`), so SIGTERM already bypasses
`atexit` today. **The shutdown lifecycle is unaffected by the server choice.**

Closing that gap would require a `signal.signal(SIGTERM, …)` handler in the
application. C5 S4 deliberately chose `atexit` and adding a signal handler is an
application change outside this audit.

### 6.8 Failure behaviour

| Event | Behaviour |
|---|---|
| Worker **thread** raises | Waitress isolates it to that request → HTTP 500 → server survives. |
| Process crash | No supervisor → process ends; supervisor (systemd/Docker) restarts; C3/C4 recovery runs. Weaker than Gunicorn's in-process worker restart. |
| SIGTERM | Immediate termination, no `atexit` (see §6.7). |
| SIGINT | Graceful, reaches `shutdown()`. |
| Reload | N/A. |
| Abnormal termination | Recovery on next start, unchanged. |

---

## 7. Flask-SocketIO compatibility analysis

### 7.1 What the current configuration is

Verified in two places, consistently:

- `ealpr/extensions.py:17` — `socketio = SocketIO(async_mode="threading", cors_allowed_origins="*")`
- `ealpr/__init__.py:131` — `socketio.init_app(app, async_mode="threading", cors_allowed_origins="*")`

`message_queue` is **never** passed anywhere (`git grep message_queue` → zero
hits), and no broker package is installed. The comment at
`extensions.py:13-16` records the reason:

> *"async_mode='threading' is required when running under the standard
> Flask/Werkzeug dev server (no eventlet/gevent installed). Without it,
> Flask-SocketIO falls back to eventlet which conflicts with the regular
> threading model and causes the whole app to freeze when any background thread
> tries to do I/O."*

### 7.2 Why an external WSGI server needs no Socket.IO change

`flask_socketio/__init__.py:249-255` (installed source):

```python
if app is not None:
    self.sockio_mw = _SocketIOMiddleware(self.server, app, socketio_path=resource)
    app.wsgi_app = self.sockio_mw
```

`init_app` wraps the Flask application itself. The returned `app` therefore
answers both ordinary routes and `/socket.io/`. **Any WSGI server that serves
`app` serves Socket.IO correctly.** This is what makes the whole exercise a
server choice rather than a rewrite.

### 7.3 Transport matrix under each candidate

| | Long-polling | WebSocket | Notes |
|---|---|---|---|
| Werkzeug (today) | ✅ | ✅ (`werkzeug.socket`) | dev server |
| Waitress | ✅ | ❌ `RuntimeError` → 500 → client stays on polling | verified from installed `simple_websocket` |
| Gunicorn `gthread` | ✅ | ✅ expected (`gunicorn.socket`) | unverified, not installed |
| Gunicorn `sync` | ✅ | ⚠️ blocks the single worker; `--timeout` risk | unsuitable |
| gevent / eventlet | ✅ | ✅ | **rejected** — needs `async_mode` change |

### 7.4 Would a message queue be required?

**No — and this must not be misread.** A message queue is required only when
more than one process emits Socket.IO events. Under both recommended
configurations there is exactly one application process, so the in-process
client registry is correct and `message_queue` stays unset. H9 is satisfied.

### 7.5 Would broadcasts remain process-local correctly?

**Yes.** `socketio.emit(...)` from `_feed_worker` and `_run_video_job` reaches
every client attached to *this* process. Since there is only this process, that
is every client. Nothing changes relative to today.

### 7.6 The concurrency trap (applies to both real candidates)

Verified from installed source:

```
engineio/socket.py:12-17
    def poll(self):
        packets = [self.queue.get(
            timeout=self.server.ping_interval + self.server.ping_timeout)]

engineio.Server defaults:  ping_interval = 25, ping_timeout = 20
```

**An idle long-polling client pins its request thread for up to 45 seconds.**

| Server | Thread supply | Idle Socket.IO clients before HTTP stalls |
|---|---|---|
| Werkzeug `threaded=True` (today) | unbounded, new thread per request | effectively unlimited (unbounded threads) |
| Waitress (default `threads=4`) | 4 | **4** |
| Gunicorn `gthread` (default `--threads=1`) | 1 | **1** |
| Gunicorn `sync` | 0 (serial) | **1 connection total** |

Under Waitress or Gunicorn with defaults, four idle dashboard tabs would
exhaust the pool, and *every* HTTP request — login, gallery, uploads — would
queue behind them until a poll returns. This is not theoretical: it follows
directly from the two verified lines above.

**A production-server migration that does not raise the thread count is an
availability regression.** See §16.

---

## 8. C4 compatibility analysis

No candidate requires a change to C4, and none is proposed.

| C4 concern | Waitress | Gunicorn `gthread w=1` | Reason |
|---|---|---|---|
| inference consumer thread | unaffected | unaffected | created inside `create_app()` in the one application process |
| supervision monitor thread | unaffected | unaffected | same |
| in-memory queue / `_queued_ids` | correct | correct | one application process |
| restart budget / self-supervision | unchanged | unchanged | thread-level, process-independent |
| startup recovery (`recover_jobs`) | runs once | runs once | one import, one process |
| `shutdown()` | unchanged | unchanged | not called by either server; reached via `atexit` |
| `atexit` hook (S4) | fires on normal exit | fires on normal exit | same process, same interpreter |
| SIGTERM | `atexit` **not** run | `atexit` expected to run (unverified) | see §10 |
| process termination | supervisor restarts the process | master restarts the worker | Gunicorn is **better** here |
| C4.1 orphan rule (`orphan_processing`) | correct | correct | both guarantee exactly one live application process at a time |
| one-consumer guarantee | structural | structural (`_ensure_worker`) | unchanged |
| Socket.IO coupling | none | none | `inference_jobs.py` has no `socketio` import |

**The one thing that must never be introduced is a second application
process.** That is already enforced for `EALPR_RELOADER`/`EALPR_WORKERS` by
`assert_single_process()`, but is **not** enforced against a server's own
`--workers`/`--reload` flags. This audit does not change that; it records the
gap (§16, §17).

---

## 9. Startup / process model analysis

### 9.1 How many times does `create_app()` run?

`app.py:22` executes it at **module scope**. The count is therefore
`(number of processes that import app) × (number of times each imports it)`.

| Scenario | `create_app()` calls | Consequence |
|---|---|---|
| `python app.py` | 1 | as today |
| `waitress-serve app:app` | **1** | correct |
| `gunicorn -w 1` (no preload) | **1** | correct |
| `gunicorn -w 1 --preload` | 1 in master **+ state inherited by forked child** | threads lost across `fork()`; `_initialization_done=True` inherited → recovery suppressed in child. **Dangerous.** |
| `gunicorn -w N` (N>1) | N | N queues, N consumers, N monitors, N recovery passes → C4.1 hazard |
| Werkzeug reloader on | 2 (parent + child) | prevented today by H3/S3 |
| Gunicorn `--reload` firing | transient ≥2 | **not** covered by S3 |

### 9.2 Parent/child and reload

- **Waitress:** neither parent/child nor reload exists. Deterministic single
  call.
- **Gunicorn:** parent/child exists by design but only the child imports the
  app; reload must be off.
- **Both:** `EALPR_*` variables are read once at import of `config.py`, so a
  mid-run environment change has no effect — same as today.

### 9.3 Import-time side effects worth knowing

Beyond `create_app()`, importing the application also runs:

- `config.py:40-41` — `SAVE_DIR.mkdir()` / `TEMP_DIR.mkdir()`
- `config.py:24-27` — two model-path warnings (now `log.warning`, S6)
- `ealpr/__init__.py` imports → `ealpr.extensions` → `SocketIO(...)` instance

All of these run once per process under every candidate. No candidate changes
them.

---

## 10. Shutdown analysis

### 10.1 The mechanism

`ealpr/inference_jobs.py:646-668`:

- `register_atexit_shutdown()` registers `_atexit_shutdown` once, guarded by
  `_atexit_registered`.
- `_atexit_shutdown()` calls the pre-existing `shutdown()` and never raises.
- `shutdown()` (`:589`) stops the monitor, sets `_shutdown`, posts `None` to
  the queue, joins the consumer, and drains in-memory ids.

It is registered from `create_app()` at `ealpr/__init__.py:238`, so whichever
process imports the application owns the hook.

### 10.2 What actually reaches it

Python runs `atexit` callbacks **only on normal interpreter shutdown**.

| Path | Reaches `shutdown()`? |
|---|---|
| `KeyboardInterrupt` (SIGINT, Ctrl-C) | **Yes** — exception unwinds the main thread, interpreter exits normally |
| Normal `return` from the server's `main()` | **Yes** |
| `sys.exit()` | **Yes** |
| **SIGTERM with no handler** | **No** — default disposition terminates the process; `atexit` never runs |
| `SIGKILL` | **No** |
| Worker/thread crash with the process still alive | **No** — hook not triggered, process keeps running |

Verified: the application installs **no** signal handlers (`signal.` → zero
hits across `ealpr/`, `app.py`, `config.py`, `models.py`), and the installed
`werkzeug/serving.py` installs none either (`serving.py:820` only catches
`KeyboardInterrupt`).

### 10.3 Conclusion on shutdown

**The shutdown lifecycle is identical before and after the server change.**

- Today: SIGINT reaches `shutdown()`; SIGTERM does not.
- Under Waitress: SIGINT reaches `shutdown()`; SIGTERM does not.
- Under Gunicorn: SIGINT reaches `shutdown()`; SIGTERM *probably* does, because
  the worker exits through `sys.exit()` — **unverified, Gunicorn not installed**.

If `shutdown()` must run on SIGTERM — which under a service manager is the
common case — the application needs a `signal.signal(SIGTERM, …)` handler that
raises `SystemExit`, letting `atexit` run. **That is an application change, is
not required by this audit, and `shutdown()` itself must not be modified.**

The safety net is unchanged either way: C3/C4 startup recovery treats an
interrupted `processing` job as orphaned and reclaims it, which is precisely
what `recover_jobs(orphan_processing=True)` was built for.

---

## 11. Security analysis

No security configuration was changed by this audit.

| Concern | Today | Under the recommendation | Notes |
|---|---|---|---|
| **Interactive Werkzeug debugger (R1)** | Enabled only when `EALPR_DEBUG=1` in development (S1); under `app.run()` Flask installs `DebuggedApplication` | **Absent.** A plain WSGI server never installs `DebuggedApplication`. Flask's `app.debug=True` can still produce detailed 500 pages, but there is no PIN-gated console. | The entry-point change alone retires R1. |
| **`allow_unsafe_werkzeug=True` (R2)** | Present at `app.py:33`; bypasses Flask-SocketIO's non-TTY production guard (`flask_socketio/__init__.py:648-653`) | **Never executed.** It sits inside `if __name__ == "__main__":`, which a WSGI server does not enter. Becomes dead code. | Removing it from `app.py` is a follow-up, not done here. |
| **Dev server as production (R3)** | Werkzeug, explicitly unsupported for production | **Retired** by the recommendation. | |
| **Host binding** | `0.0.0.0:5000` hard-coded in `app.py:26-27` | **Not applied** — the `__main__` block does not run. The listener is whatever the server's `--listen` says. | Must be set explicitly to `127.0.0.1` behind a reverse proxy; do not rely on a default. |
| **Forwarded headers / proxy** | Not configured anywhere | Unchanged — still not configured | Behind a reverse proxy the app will see the proxy's IP and `http` scheme. Flask's session cookie `Secure` flag and any future client-IP logging would be wrong without `ProxyFix`. **Gap, unchanged by this audit.** |
| **Production secret validation (S5)** | Runs in `create_app()` before the Flask object exists | **Unaffected** — runs identically under any server that imports `app` | No change needed. |
| **Single-process enforcement (S3)** | `assert_single_process()` reads `EALPR_RELOADER` / `EALPR_WORKERS` | Waitress: **no conflicting flag exists** → guard sufficient. Gunicorn: **`--workers` and `--reload` are invisible to the guard** → guard insufficient. | A decisive argument for the recommendation. |
| **Debug outside development (S1)** | Refused at startup | Unaffected — `resolve_runtime_mode()` still runs first | |
| **`cors_allowed_origins="*"`** | Set twice (`extensions.py:17`, `__init__.py:131`) | Unchanged | Permissive; worth revisiting if the app is ever exposed beyond a same-origin front end. Not changed here. |
| **Rate limiting (R10)** | None in the application | Unchanged | Pre-existing gap, out of scope. |

---

## 12. Performance considerations

No benchmark was run, nothing was installed, and **no performance improvement
is claimed**. The following is analysis of the concurrency model only.

### 12.1 Concurrency model

- **Inference is serial by design.** `ealpr/ai_models.py:15`
  `INFERENCE_LOCK = threading.RLock()`, acquired by `predict_with_lock`
  (`:96-98`) around every `model.predict`. One inference runs at a time
  regardless of how many processes or threads exist. The C2 header
  (`inference_jobs.py:10-30`) records this as the reason a broker and a worker
  fleet were rejected: *Celery would buy concurrency the lock forbids.*
- **One process / one worker is therefore not the inference bottleneck** — the
  lock is. Adding processes would multiply model memory without adding
  inference throughput, and would break C4 (§8).
- **HTTP request handling** is mostly short: page rendering, JSON polling for
  job status, image serving. The long-running work (`_run_video_job` at
  `detection.py:241`, `_feed_worker` at `utils.py:510`) already runs in
  **background threads outside the request thread**, so no request is held for
  the duration of a video or a camera stream. That is a helpful property: it
  keeps request threads free.
- **Socket.IO is the exception.** Long-polling holds a request thread for up to
  45s per idle client (§7.6).

### 12.2 Concurrency risk by candidate

| Candidate | Threads available | Risk |
|---|---|---|
| Werkzeug (today) | unbounded | No starvation, but unbounded thread growth is itself a risk under load. |
| **Waitress** | `threads=` (default 4) | **Must be raised.** 4 idle Socket.IO clients stall all HTTP. |
| Gunicorn `gthread` | `--threads=` (default 1) | **Must be raised**, same reason, worse default. |
| Gunicorn `sync` | 1 serial | Unusable: one WebSocket or one idle poll blocks everything. |

### 12.3 Blocking inference interaction

`INFERENCE_LOCK` blocks the *inference* thread, never a request thread — the
async job path hands work to the consumer thread and returns a `job_id`
immediately. Synchronous `/process_plate` does block its request thread for the
inference duration (~2.4s measured in C2), which is true under every candidate
and is unaffected by the server choice.

### 12.4 Throughput ceiling

One process, one worker, one lock. That is a hard ceiling and it is
**deliberate** (H1–H7). No candidate changes it, and none should claim to. The
choice between candidates is about *correctness of transport and process
model*, not about requests per second — and this audit made no measurement.

### 12.5 Socket.IO threading

`async_mode="threading"` means engineio spawns a `DaemonThread`
(`engineio/async_drivers/threading.py:6-9`) per WebSocket. Under long-polling
no such thread is created. Either way these are cheap threads inside the single
process, unaffected by the server choice.

---

## 13. Recommended option

### **Waitress**

**Why it fits EALPR specifically — not because it is popular:**

1. **It is the only candidate that satisfies all nine hard constraints
   simultaneously.** One process, one worker, no reloader, threaded
   `async_mode`, no broker, and no server-side flag that can silently contradict
   `assert_single_process()`.
2. **It does not need `async_mode` changed.** It is a non-monkey-patching,
   thread-based server — the exact model `ealpr/extensions.py:13-16` says this
   application depends on.
3. **It is a production server**, so adopting it retires audit findings R2
   (`allow_unsafe_werkzeug`) and R3 (development server in production) without
   touching any application logic.
4. **It runs on the platform this project actually occupies.** Gunicorn does
   not, and no deployment target has been established anywhere in the repository
   — there is no Dockerfile, Procfile or systemd unit.
5. **Its failure mode is a transport downgrade, not a correctness break.** The
   WebSocket upgrade fails and Engine.IO keeps the client on long-polling; every
   `live_detection` event is still delivered, and `message_queue` is still not
   required. No C4, queue, recovery or Socket.IO behaviour changes.
6. **It has no reloader and no `--workers`**, so the two latent
   single-process escape hatches Gunicorn offers are structurally closed.

**The exact risks this recommendation introduces are in §17 and are real.**
The two that matter are the WebSocket-to-polling downgrade and the mandatory
thread-pool sizing.

---

## 14. Second-best option

### **Gunicorn with `--workers 1 --worker-class gthread` (Linux only)**

Chosen as second-best because:

- It preserves the **WebSocket** transport, which Waitress cannot.
- It gives **worker auto-restart on crash**, which Waitress cannot — a crashed
  worker re-imports `app` and re-runs C3 recovery, a genuine robustness gain.
- Its SIGTERM path *probably* reaches `atexit` (unverified), which would make
  `shutdown()` run under a service manager where Waitress does not.

It is **not** recommended first because:

- It needs **two OS processes** (master + worker), a literal deviation from H1,
  even though the master runs no application code.
- **`assert_single_process()` cannot see `--workers` or `--reload`**, so C5's
  S3 guard would be silently bypassed by a plausible misconfiguration.
- **It cannot run on Windows**, the platform this project currently uses, and
  no Linux deployment target is documented anywhere.
- `--preload` and `--reload` are both live foot-guns for C4 (§5.1), and
  `--worker-class sync` (the default) is actively unusable here.
- It could not be verified in this environment: **Gunicorn is not installed and
  installing it was forbidden.** Every Gunicorn statement in this report is
  either read from `simple_websocket`'s installed source or marked unverified.

If the application owner later establishes a Linux deployment target and
requires WebSocket rather than long-polling, this option should be
re-evaluated, and the `EALPR_WORKERS` guard gap must be closed at the same
time.

---

## 15. Rejected options

| Option | Verdict | Exact reason |
|---|---|---|
| **Gunicorn + `--worker-class sync` (the default)** | **Rejected** | A sync worker serves one request at a time. One WebSocket client, or one idle long-poll, occupies the single worker indefinitely; the app would serve no HTTP. Gunicorn's default `--timeout 30` heartbeat is updated only between requests, so the worker would also be killed while a client watches the feed. Unusable without changing worker class. |
| **Gunicorn + `gevent`, or Gunicorn + `eventlet`, or `eventlet`/`gevent` async_mode** — the recipe Flask-SocketIO's documentation traditionally recommends | **Rejected outright** | (a) It requires changing `async_mode` from `threading`, which violates **H8** and is a Socket.IO code change this audit may not make. (b) `ealpr/extensions.py:13-16` records that this project's own reasoning is that without `threading` it "falls back to eventlet which **conflicts with the regular threading model and causes the whole app to freeze** when any background thread tries to do I/O." (c) Greenlet monkey-patching would interact with `INFERENCE_LOCK`, the daemon consumer/monitor threads, and the `atexit` hook — all of which are C4 mechanisms. (d) Neither package is installed. |
| **Status quo: `python app.py` on Werkzeug, as the production entry point** | **Rejected for production** | It is Werkzeug's development server. It keeps `allow_unsafe_werkzeug=True` (R2), keeps the development server as production (R3), and `flask_socketio/__init__.py:648-653` shows the flag is *required* whenever stdin is not a TTY — i.e. under systemd, Docker or `nohup`, removing it makes the app refuse to start. It remains correct as the **development** command. |
| **`allow_unsafe_werkzeug` removal as a substitute for S2** | **Rejected** | Not a fix: it would raise `RuntimeError` under any service manager. The flag is a symptom; the server is the cause. |
| **ASGI servers (uvicorn, hypercorn, daphne) and any Flask→Quart/ASGI migration** | **Not evaluated** | The application is a synchronous **WSGI** Flask application. The installed Flask-SocketIO `run()` offers no ASGI path for `async_mode="threading"`, and an ASGI migration would be a framework change, not a server selection. The source does not make it relevant. |
| **Any message broker (Redis, Celery, RQ, Kafka, RabbitMQ)** | **Rejected** | Violates **H9**. Also unnecessary: a broker is only needed for multi-process Socket.IO, and multi-process is the very thing this audit refuses to recommend. |
| **Multiple workers behind a load balancer** | **Rejected** | Violates **H1, H2, H4–H7**. Per the C5 audit §10.3, two application processes let one process's startup recovery fail a job the other is actively processing, and strand `queued` jobs. C4.1's orphan rule is only correct with exactly one live process. **Any multi-worker architecture is explicitly incompatible with the current C4 design.** |

---

## 16. Exact implementation plan for the next block

Nothing below has been performed. It is the plan for a future, authorised S2
implementation block.

### 16.1 Dependency changes

1. Add **one** line to `requirements.txt`: `waitress` (pin a specific version
   at implementation time; do not leave it unpinned — R13 from the C5 audit).
2. No other dependency. No broker, no gevent, no eventlet.

### 16.2 Required configuration changes

| Item | Requirement | Why |
|---|---|---|
| Thread pool | `waitress-serve --threads=<N>` where **N ≥ expected concurrent Socket.IO clients + HTTP headroom** (start at 50–100) | §7.6 — 45s per idle poll; Waitress's default is 4 |
| Listener | `--listen=127.0.0.1:5000` (or equivalent) | Never expose directly; put a reverse proxy in front |
| `EALPR_ENV` | `production` | S1/S5 |
| `SECRET_KEY` | unique random value | **S5 refuses startup otherwise** |
| `DEFAULT_ADMIN_PASSWORD` | non-default value | **S5 refuses startup otherwise** |
| `EALPR_WORKERS` | `1` or unset | S3 |
| `EALPR_RELOADER` | off (default) | S3 |
| Reverse proxy | nginx/Caddy in front, TLS terminated there | Deployment change; none exists today |
| Forwarded headers | `ProxyFix` (or equivalent) if client IP / scheme must be correct | Currently absent — **optional, separate decision** |

### 16.3 Required entry-point change

- Production command becomes a Waitress invocation of the WSGI callable that
  already exists at `app.py:22` (e.g. `waitress-serve … app:app`).
- **No new `wsgi.py` is strictly required**, because `app = create_app()` is
  already at module scope. Adding a thin `wsgi.py` is still worth considering
  purely for naming clarity and to avoid any future module/attribute ambiguity
  with the `app:app` form — an implementation-time judgement call.
- `python app.py` remains the **development** command, unchanged.

### 16.4 Required code changes (future, not now)

| File | Change | Reason |
|---|---|---|
| `app.py` | Retire `socketio.run(...)`, `allow_unsafe_werkzeug=True`, `host`/`port` | Dead under a WSGI server; retires R2 |
| `requirements.txt` | add `waitress` | §16.1 |
| `README.md` | replace "the only supported startup command is `python app.py`" with dev vs production commands | Currently states the opposite |
| `.env.example` | document the production command | same |
| `tests/test_secret_logging_docs.py` | **`test_no_wsgi_server_is_documented_as_available` must be updated** | It asserts `README.md` and `.env.example` contain **none** of `waitress`, `gunicorn`, `gevent`, `eventlet`, `uvicorn`, `hypercorn`. Documenting the chosen server **will fail this test unless it is changed in the same block.** This is a hard sequencing constraint on the implementation. |
| `tests/test_secret_logging_docs.py` | `test_documents_state_the_startup_command`, `test_no_production_readiness_is_claimed` | Same reason — both assert S2 is *not* implemented |
| New tests | assert `create_app()` is invoked exactly once under the server entry point; assert `threads` configuration; assert `assert_single_process()` still passes | Regression coverage |

**Deliberately unchanged:** `ealpr/inference_jobs.py` (C4), all Socket.IO code,
`async_mode`, `message_queue`, worker count, AI/OCR, UI.

### 16.5 Documentation changes

- README *Deployment status* section must be rewritten: S2 implemented, server
  named, `threads` sizing explained, long-polling expectation stated, and the
  "not production-ready" sentence revisited (other gaps — rate limiting,
  `ProxyFix`, dependency pinning — will still justify keeping it).
- Add the reverse-proxy and signal-handling guidance (§10.2, §11).

### 16.6 Verification plan for that block

1. Full suite green (currently **210 passed, 2 skipped**) after updating the
   three documentation tests.
2. `git diff` shows only the files listed in §16.4.
3. Read-only `_id`-sorted integrity snapshot unchanged:
   `9200c7588833bb5456221036de1d6913`.
4. Subprocess boot through the new entry point in `EALPR_ENV=production` with
   valid secrets → starts, logs `Starting EALPR in production mode`.
5. Confirm `create_app()` executes exactly once (one log line, one recovery
   summary).
6. Confirm Socket.IO long-polling connect and `live_detection` delivery with a
   real client.
7. Confirm WebSocket upgrade fails gracefully and the client stays on polling —
   **expected**, and it must be recorded as accepted behaviour rather than
   treated as a defect.
8. Confirm `assert_single_process()` still passes with the documented settings.
9. Confirm no broker package was installed.

---

## 17. Risks

Risks **introduced** by the recommendation:

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **N1** | **WebSocket transport unavailable.** Every client falls back to long-polling; `live_detection` still arrives but with an extra round-trip per packet and a 500 logged on every upgrade probe. | **High** (visibility), Medium (function) | Document as accepted. If WebSocket is a hard requirement, use the second-best option on a Linux target instead. |
| **N2** | **Thread-pool exhaustion.** 4 idle Socket.IO clients pin 4 threads for up to 45s; Waitress's default `threads=4` means total HTTP stall. | **Critical** | `--threads` sized above expected concurrent clients. Not optional. |
| **N3** | **SIGTERM bypasses `atexit`**, so `shutdown()` does not run under a service manager. In-flight job left `processing`. | Medium (hygiene) | Unchanged from today; C3/C4 recovery covers correctness. Add a SIGTERM handler in a later block if operator visibility is required. |
| **N4** | **No worker supervision.** Waitress has no master; if the process dies, only an external supervisor restarts it. Gunicorn would restart a crashed worker in-process. | Medium | Run under systemd/Docker with restart policy. |
| **N5** | **Documentation tests fail** the moment the server is named in `README.md`/`.env.example`. | Low (process) | Update those tests in the same block — §16.4. |
| **N6** | **`--listen` default misconfiguration** could expose the app directly or on an unexpected port. | Medium | Always set `--listen` explicitly; front with a reverse proxy. |

Risks **reduced** by the recommendation:

- **R1** (interactive debugger on a network interface) — retired: a WSGI server
  never installs `DebuggedApplication`.
- **R2** (`allow_unsafe_werkzeug`) — retired: the flag is inside the
  `__main__` block and never executes.
- **R3** (development server as production) — retired.
- **R4** (one-token reloader edit creates a dual-process deployment) —
  Waitress has no reloader at all, so the hazard is structurally removed rather
  than guarded.

Risks **carried over unchanged** (pre-existing, out of scope here):

- R10 rate limiting absent; `ProxyFix` absent; dependency pinning (R13);
  `cors_allowed_origins="*"`; the C4 monitor-has-no-supervisor gap (R16);
  `assert_single_process()`'s blindness to server-level flags.

Risks **if the second-best option were chosen instead**:

- Two OS processes (deviation from H1).
- `--workers` and `--reload` invisible to `assert_single_process()` → S3 guard
  bypassable.
- `--preload` breaks C4 thread inheritance across `fork()`.
- Cannot run on Windows.
- Unverified — Gunicorn is not installed.

---

## 18. Explicit statement

> ### S2 implementation was NOT performed.
>
> No production WSGI server was installed, configured, chosen for use, or run.
> No package was installed and no package was uninstalled.
> `requirements.txt` was not modified.
> No application code was modified: `app.py`, `config.py`,
> `ealpr/__init__.py`, `ealpr/inference_jobs.py`, `ealpr/extensions.py`, all
> Socket.IO code, all blueprints, all AI/OCR code and all UI are byte-identical
> to commit `dcd2e601`.
> No test was modified or added.
> No WSGI entry point, Dockerfile, docker-compose, Procfile, systemd unit or
> CI/deploy workflow was created.
> Worker count was not changed. C4 was not changed. The shutdown lifecycle was
> not changed.
> Nothing was committed. No database write, no inference job, no detection, no
> visitor and no configuration mutation occurred.
>
> **The only file created by this audit is `PHASE2_BLOCK_C5_S2_SERVER_AUDIT.md`.**

---

## Appendix — Verification performed (read-only)

| Check | Type | Result |
|---|---|---|
| `git log --oneline -1` | Git | `dcd2e601` |
| `git status --porcelain` before/after | Git | only the 9 pre-existing tracked `.pyc` byte-changes + this report |
| Full read of `app.py`, `config.py`, `ealpr/__init__.py` | Read | — |
| Read of `ealpr/inference_jobs.py` (header, `shutdown`, `atexit` block) | Read | — |
| Read of `ealpr/extensions.py`, `requirements.txt`, `README.md` | Read | — |
| Read of `PHASE2_BLOCK_C5_S5_S6_S7_REPORT.md`, `PHASE2_BLOCK_C5_S1_S3_S4_REPORT.md`, `PHASE2_BLOCK_C5_AUDIT_REPORT.md` | Read | — |
| Read of installed `flask_socketio/__init__.py:190-255`, `:600-661` | Read | — |
| Read of installed `engineio/socket.py`, `server.py`, `async_drivers/threading.py`, `async_drivers/_websocket_wsgi.py` | Read | — |
| Read of installed `simple_websocket/ws.py:250-324` | Read | **decisive finding** |
| Read of installed `werkzeug/serving.py` signal/KeyboardInterrupt handling | Read | no signal handlers |
| `importlib.metadata` version query for 16 packages | Dependency | 6 present, 10 **ABSENT** (no server, no broker) |
| `engineio.Server` signature inspection for `ping_interval`/`ping_timeout`/`allow_upgrades` | Runtime (read-only) | `25`, `20`, `True` |
| Socket.IO emit/handler sites enumerated across `ealpr/`, `templates/` | Static | 2 emitters, 2 handlers, 1 client (`io()`) |
| `git grep` for `signal.`, `atexit`, `message_queue`, `_run_video_job`, `INFERENCE_LOCK` | Static | as reported |
| Read-only `_id`-sorted integrity snapshot | MongoDB (read-only) | **unchanged** — `9200c7588833bb5456221036de1d6913`, 66 / 1023 / 0, 6,355,490 + 946,803, 812 / 211, 317 active |

**Explicitly not run:** any `create_app()` invocation, the test suite, any
server startup, any port binding, any package installation, any `pip` command,
any MongoDB write, and any multi-process deployment.

**Gunicorn and Waitress are not installed**, so every statement about their
internal behaviour is either (a) read from the installed `simple_websocket`
source that interfaces with them, or (b) marked **unverified** in the text.

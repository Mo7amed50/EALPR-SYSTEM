# Phase 2 / Block C5 — S2 Implementation Report

**Production WSGI server (Waitress) — implementation and verification**

| Field | Value |
|---|---|
| Base commit | `dcd2e601` — *phase2: harden production secrets logging and docs* |
| Scope | Block C5 **S2 only** (production WSGI server selection, install, config) |
| Dependency added | `waitress==3.0.2` (single line in `requirements.txt`) |
| Entry point | WSGI callable `app:app` (existing `app = create_app()` at module scope) |
| Commit | `feat(server): add Waitress production entry point` (exactly one) |
| Tests | Full suite: **232 run, OK, 2 skipped** (`unittest discover`, ~600 s) |
| Data integrity | Baseline unchanged (digest `9200c7588833bb5456221036de1d6913`, 66 / 1023 / 0) |
| Status | Implemented, verified. Do not claim full production readiness. |

---

## 1. Scope

Phase 2 / Block C5 **S2** — *production WSGI server selection* — was the one
remaining item of Block C5 (S1, S3, S4, S5, S6, S7 were committed previously).
This block selects, installs and documents **Waitress** as the production WSGI
server, and adds tests that verify the selection is real and that the existing
single-process / Socket.IO / C4 design is preserved.

**Explicitly not done** (out of scope, see §9): multi-worker, external broker,
gevent/eventlet, async-mode change, Docker/Procfile/systemd/CI, reverse proxy
config, rate limiting, load testing.

---

## 2. Why Waitress

The audit (`PHASE2_BLOCK_C5_S2_SERVER_AUDIT.md`) compared the realistic options
against the hard constraint that has governed Block C5 from the start: the
system **must run as a single process** running exactly one inference consumer
(`assert_single_process()`), with `flask-socketio` in `async_mode="threading"`.

| Candidate | Verdict |
|---|---|
| **Waitress** | Pure-Python, threaded WSGI server. Single process, multi-threaded. No fork/preload model. Matches `threading` async mode. Windows-native (no fcntl/fork). **Selected.** |
| Gunicorn | Pre-fork, POSIX-only, multi-process worker model conflicts with the single-process/consumer rule. |
| uWSGI | Heavier, POSIX-centric, worker model again conflicts. |
| gevent/eventlet workers | Would change the Socket.IO async mode and pull in monkey-patching. |

Waitress's threading model is the same concurrency model the application already
uses, so it adds a production HTTP server **without** changing the concurrency
architecture.

---

## 3. Dependency

`requirements.txt` gained exactly one line, after `flask-socketio`:

```
waitress==3.0.2
```

- `3.0.2` was the current release at implementation time (`pip index versions
  waitress`).
- Pinned to an exact version, consistent with the rest of the file.
- No transitive server/broker packages were introduced. Verified still absent:
  `gunicorn`, `gevent`, `eventlet`, `redis`, `celery`, `rq`.
- Note: `waitress` does not expose a `__version__` attribute; version is read
  via `importlib.metadata.version("waitress")` (used by the tests).

---

## 3a. Entry point

`app.py` already exposes the application at module scope:

```python
app = create_app()
```

so the WSGI callable is `app:app`. **`app.py` was not modified.** Keeping
`python app.py` as the development command and `app:app` as the production
target means there is exactly one application factory and one assembly path.

Production invocation:

```powershell
waitress-serve --listen=127.0.0.1:5000 --threads=50 app:app
```

---

## 3b. Thread-pool sizing

`--threads=50` is set explicitly (Waitress defaults to 4).

Waitress threads are shared by all concurrent work, and **an idle Socket.IO
long-polling request pins one thread for up to ~45 s** at a time
(`ping_interval` 25 s + `ping_timeout` 20 s — visible in the handshake as
`pingInterval: 25000, pingTimeout: 20000`). With the default of 4 threads, four
idle clients would exhaust the pool and stall every other request (including
the UI). The audit recommended starting in the 50–100 range; **50** is used here
as the conservative, explicitly documented initial value.

This is a **starting configuration, not a benchmarked value**. The README says
so explicitly. It can be raised without code changes.

---

## 4. Documentation changes

- **`README.md`** — the "Deployment status" section was rewritten: S2 is now
  documented as implemented with Waitress 3.0.2; the development command
  (`python app.py`) and the production command
  (`waitress-serve --listen=127.0.0.1:5000 --threads=50 app:app`) are listed;
  the thread-count rationale is stated; the Socket.IO **long-polling** behaviour
  is documented; the lack of built-in process supervision and the `atexit`/SIGTERM
  limitation are documented; the application is stated to be **not fully
  production-ready**.
- **`.env.example`** — the startup-command comment now shows both the
  development and the production invocations.

No other documentation was touched.

---

## 5. Socket.IO transport behaviour under Waitress

This is the single most important behavioural finding, and it is verified
end-to-end, not assumed.

- The Socket.IO **HTTP long-polling** handshake succeeds:
  `GET /socket.io/?EIO=4&transport=polling` returns **200** and advertises
  `"upgrades":["websocket"], "pingTimeout":20000, "pingInterval":25000`.
- A real python-socketio client connects with `transport() == "polling"` and
  receives an emitted `live_detection` event.
- A **raw WebSocket upgrade attempt** (`transport=websocket` + `Upgrade:
  websocket` headers) returns **HTTP 500**. The server traceback ends with:

  ```
  simple_websocket/ws.py:290  raise RuntimeError(
      'Cannot obtain socket from WSGI environment.')
  ```

  Waitress does not expose a raw socket in the WSGI environ, so the
  `simple_websocket` path cannot complete a WebSocket handshake.

**Consequence, documented in the README:** Socket.IO clients negotiate down to
**HTTP long-polling**. All `live_detection` events are still delivered; the
WebSocket transport is simply unavailable under this server. This is why §3b
sizes the thread pool generously. The Socket.IO **client-side transport
configuration was not changed** (out of scope); the downgrade is a property of
the server, and clients fall back automatically.

---

## 6. Tests added

New file **`tests/test_wsgi_server.py`** (22 tests):

| Class | Covers |
|---|---|
| `WaitressDependencyTests` | `waitress==3.0.2` pinned in `requirements.txt`; importable; no broker/multi-worker package added. |
| `ProductionDocumentationTests` | README + `.env.example` document Waitress, the `app:app` target, the loopback bind, an explicit `--threads=N`, long-polling, the absence of WebSocket, external supervision, and "not fully production-ready". |
| `GuardRegressionTests` | `assert_single_process()`, `resolve_runtime_mode()`, `assert_production_secrets()` still reject multi-worker / default-secret production configs (C5 guards intact). |
| `EntryPointInitializationTests` | Importing/launching the WSGI entry point initializes the application **exactly once** (one startup log line, no duplicate consumers). |
| `WaitressSmokeTests` | Real subprocess boot in production mode through the WSGI callable (`app:app`) in a fresh process; HTTP served. Uses **per-run, ephemeral** `SECRET_KEY`/`DEFAULT_ADMIN_PASSWORD` generated with `secrets.token_hex` (never persisted, never printed). |
| `SocketIOIntegrationTests` | Long-polling handshake 200 + advertised upgrade; client connects as polling and receives `live_detection`; raw WebSocket upgrade returns 500 (documents the limitation). |

Driver scripts are written to the OS temp directory via `tempfile`, never into
the repository.

Existing test **`tests/test_secret_logging_docs.py`** was updated because it
previously asserted the *absence* of S2:

- `test_documents_state_that_s2_is_not_implemented` →
  `test_documents_state_that_s2_is_implemented`.
- `test_no_production_readiness_is_claimed` →
  `test_mentions_production_readiness_limitations` (asserts "not fully
  production-ready").
- `test_no_wsgi_server_is_documented_as_available` →
  `test_wsgi_server_is_documented` (now requires `waitress`; still forbids
  `gunicorn`/`uvicorn`/`hypercorn`/`gevent`/`eventlet`).

---

## 7. Verification performed

All commands run from `D:\EALPR SYSTEM` with `.venv\Scripts\python.exe`.

1. **Pre/post integrity snapshot** (read-only, `_id`-sorted):
   `det_count=66`, `visitor_count=1023`, `inference_job_count=0`,
   `content_digest=9200c7588833bb5456221036de1d6913`, `path_backed=6`,
   `authorized=812`, `unauthorized=211`, `active_visitors=317`.
2. **Targeted** — `tests.test_wsgi_server` (22 tests) → **OK**.
3. **Targeted** — `tests.test_secret_logging_docs` + wsgi dependency/doc/guard
   classes → **OK** (45 tests).
4. **Full suite** — `python -m unittest discover -s tests -t .` →
   **Ran 232 tests … OK (skipped=2)** in ~600 s.
5. **Integrity re-check** after the full suite → digest and counts unchanged
   (item 1 values).

Note: `test_inference_jobs` alone takes ~480 s; runs must use a shell timeout
well above the default (a killed run can leave orphaned test artifacts — see §8).

---

## 8. Data integrity

The full suite (including the new subprocess-booting tests) left the MongoDB
`ealpr` state byte-for-byte identical to the baseline:

- `content_digest = 9200c7588833bb5456221036de1d6913` (unchanged)
- `DetectionResult = 66`, `Visitor = 1023`, `InferenceJob = 0` (unchanged)

During development, interrupted `test_inference_jobs` runs (killed by a too-short
shell timeout) transiently left `__c2_test_*` detection rows and `c2test_*`
inference jobs. These were removed by exact-`_id` deletion and the baseline was
restored before any commit; the final full run (item 4/5 above) left **no**
residue.

---

## 9. Explicitly unchanged / out of scope

- **C4 untouched:** `ealpr/inference_jobs.py` (monitor lifecycle, atomic job
  claim, atomic recovery), `ealpr/ai_models.py`, the single `INFERENCE_LOCK`,
  and the Socket.IO extension wiring were **not modified**.
- No multi-worker server, no external broker (Redis/Celery/RQ/Kafka/RabbitMQ),
  no gevent/eventlet, no `async_mode` change, no reloader.
- `python app.py` remains the development entry point; one app factory
  (`ealpr/__init__.py:create_app`); no alternate factory, no duplicated init.
- No Dockerfile, Procfile, systemd unit, or CI change.
- No Socket.IO **client** transport change.
- No schema, model, or migration change.

---

## 10. Known limitations / residual risks

1. **No WebSocket under Waitress** — clients use HTTP long-polling (§5). Higher
   per-request overhead; thread pool sized accordingly.
2. **No process supervision** — restart on crash is the operator's responsibility
   (systemd / Windows Service / Docker restart policy). Documented.
3. **`atexit` may be bypassed on SIGTERM** — unchanged from the prior design;
   C4 recovery reconciles in-flight jobs on the next start.
4. **Thread count is not benchmarked** — 50 is an initial value; load testing
   remains open work.
5. **Still not fully production-ready** — rate limiting, `ProxyFix` (if behind a
   proxy that affects client IP/scheme), dependency hardening, monitoring, and
   load testing remain. Neither the README nor this report claims otherwise.

---

## 11. Rollback

Revert the single commit. No schema, data, or migration was changed. Waitress is
additive: the development entry point (`python app.py`) is unaffected, so removing
the dependency and the documentation reverts cleanly to the pre-S2 state.

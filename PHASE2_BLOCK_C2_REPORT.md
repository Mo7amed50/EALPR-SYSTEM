# Phase 2 — Block C2: Asynchronous Inference Job Architecture

**Status:** Complete
**Block:** C2 — Async inference jobs
**Baseline commit:** `3eec1777` (Block C1)
**Commit:** `phase2: add asynchronous inference job architecture`

**Outcome:** A background worker executes plate inference off the request
thread. The synchronous `/process_plate` path is untouched and remains the
compatibility path.

---

## 1. Architecture before C2

`POST /process_plate` (`ealpr/blueprints/detection.py:87`) did everything
inline, on the Flask request thread:

```
request thread
  ├─ validate upload
  ├─ save upload to TEMP_DIR
  ├─ get_plate_detection_model() / get_ocr_model()      # lazy load on first use
  ├─ process_license_plate(temp_path, plate, ocr)        # ~2.4s warm, ~8.5s cold
  │     └─ predict_with_lock() -> INFERENCE_LOCK         # global RLock
  ├─ Visitor.objects(license_plate=...).first()
  ├─ visitor.entry_datetime_utc = now; visitor.save()   # side effect
  ├─ save_detection_images(...)                         # disk write
  ├─ DetectionResult(...).save()
  └─ jsonify(response)   # includes base64 processed image
```

A 2.4s (or 8.5s cold) inference occupied a request thread for its whole
duration. `INFERENCE_LOCK` is a global `RLock` (`ealpr/ai_models.py:15`), so
concurrent inference is already impossible by design — the second request
waited on the lock *and* on a busy request thread.

The only pre-existing async work was `_run_video_job` (`detection.py:241`): a
`threading.Thread` + `app.app_context()` with an in-memory `_video_jobs` dict.
That is the pattern C2 follows.

### C2.0 audit findings

| Question | Finding |
| --- | --- |
| One plate inference | `process_license_plate(image_path, plate_model, ocr)` — `ealpr/utils.py:141`, delegating to `_detect_on_array` (`utils.py:157`). Takes a **file path** and takes `INFERENCE_LOCK` internally. |
| Persists DetectionResult | **No such function exists.** Persistence is inlined in the route at `detection.py:187-198`, interleaved with the visitor lookup and the `entry_datetime_utc` side effect. |
| Thread-safe state | `INFERENCE_LOCK` (RLock) serialises inference. Model globals are lazily loaded by `_load_yolo_models` with **no lock** on the `_models_loaded` check — a latent double-load race. |
| Worker-callable unchanged? | Yes. It needs only a path and already locks itself. |
| Server | Werkzeug **dev** server via `socketio.run(..., use_reloader=False)`, single process, threaded. No gunicorn/waitress. |

Two consequences shaped the design:

- **`current_user` is a request-local proxy.** It does not exist in a worker
  thread, so `processed_by` must be captured at enqueue time and passed to the
  worker. `DetectionResult.processed_by` still ends up holding the same user.
- **Single process means one YOLO instance.** Dispatching work to other
  *processes* would load a second model.

---

## 2. Worker technology selected, and why

**Installed queue infrastructure: none.** No Redis, Celery, RQ, Flask-Executor,
Huey, or arq. No gunicorn or waitress either.

**Selected: a single in-process consumer thread + `queue.Queue` + durable job
documents in MongoDB.** No new dependency.

Celery was considered and rejected on evidence, not preference:

1. **`INFERENCE_LOCK` already caps inference at one at a time.** A worker fleet
   would queue on the same lock and still execute serially. Celery would buy
   concurrency the lock forbids — a broker, a new failure mode, and operational
   surface for **zero** throughput gain.
2. **The app is one process.** Moving work to separate processes would load a
   second YOLO instance per worker, violating the C2.9 constraint against one
   model instance per job and wasting GPU memory.
3. **The pattern already exists in this codebase.** `_run_video_job` runs in a
   `threading.Thread` inside `app.app_context()`. Reusing it introduces no new
   operational concept for the operator.

Job state lives in Mongo, not a process-local dict, so status survives a
restart. This is a deliberate improvement over the existing `_video_jobs` store,
which loses all state on restart and has **no ownership check at all**.

---

## 3. New job representation

`InferenceJob` in `models.py` — a new, isolated `inference_job` collection.
`DetectionResult` and `Visitor` are **not** modified. No migration; no existing
record is touched. Every field is documented inline in the model.

| Field | Purpose |
| --- | --- |
| `job_id` | UUID4 hex, unique, indexed. The only identifier exposed over HTTP. |
| `user` | Owning user; enforces ownership and populates `processed_by`. |
| `status` | `queued` \| `processing` \| `completed` \| `failed`. |
| `image_path` | Staged upload on the server. **Never serialised into any response.** |
| `original_filename` | Sanitised client filename, display only. |
| `created_at` / `started_at` / `completed_at` | Full lifecycle timing. |
| `detection` | Reference to the `DetectionResult` on success. |
| `plate_text` | Recognised plate, for a cheap status read. |
| `error_code` / `error_message` | Stable category + **sanitised** text. |
| `attempts` | Worker attempt count, diagnostics. |

The full exception is written to the **server log only**. It is never stored,
so a failed job cannot leak internals through the status endpoint.

---

## 4. Worker lifecycle

`ealpr/inference_jobs.py`.

```
POST /api/inference/jobs
  ├─ validate (extension, size, filename)  ── same rules as /process_plate
  ├─ model availability gate               ── 503 if models missing
  ├─ stage upload to TEMP_DIR
  └─ enqueue(): save InferenceJob(status=queued), push id, start consumer
     └─ return 202 {success, job_id, status: "queued"}     [no inference]

consumer thread (one, daemon)
  └─ run_job(app, job_id)
       ├─ _claim(job_id)      atomic queued -> processing, $inc attempts
       ├─ get_plate_detection_model() / get_ocr_model()
       ├─ process_license_plate(path, plate, ocr)   <-- EXISTING function
       ├─ cv2.imencode processed image
       ├─ visitor lookup + entry_datetime_utc side effect  (mirrors process_plate)
       ├─ save_detection_images() + DetectionResult(...).save()
       ├─ completed + detection reference
       └─ finally: delete the staged temp file
     on any controlled failure: _fail(code) -> failed + safe message
     on any unexpected exception: caught, logged with traceback, job = failed
```

The consumer thread wraps `run_job` in its own `try/except`, so a bug in the
job handler cannot kill the thread and silently stop all future processing.

### The claim is atomic

The first version loaded the job, checked the status, and saved — a
read-then-act race. Two runners could both read `queued` and both run inference,
producing a **duplicate DetectionResult**. The claim is now a single
`findAndModify` filtered on `status='queued'`, so exactly one runner can win.
`test_concurrent_runners_cannot_both_claim_a_job` fires four threads at one job
and asserts exactly one wins.

This was not theoretical: it reproduced while writing the smoke test, when the
auto-started worker and a manual `run_job` both claimed the same job, and the
loser then failed because the winner had already deleted the temp file.

---

## 5. New endpoints

### `POST /api/inference/jobs` → **202 Accepted**

```json
{ "success": true, "job_id": "1f22ee8e...", "status": "queued" }
```

Auth required (`login_required`), DB required (`db_required`). Performs **no**
inference — `test_post_does_not_perform_inference_in_the_request` asserts the
inference function is never called during the request.

### `GET /api/inference/jobs/<job_id>`

```json
{
  "success": true,
  "job_id": "1f22ee8e...",
  "status": "completed",
  "created_at": "2026-09-28T12:00:00Z",
  "started_at": "2026-09-28T12:00:01Z",
  "completed_at": "2026-09-28T12:00:05Z",
  "detection_id": "6ab995aa7dcbc406d80633b0"
}
```

`/process_plate`, `/process_video`, the camera stream and Socket.IO are
untouched (C2.7). Only the reusable job/worker boundary was built.

---

## 6. Job ownership and security

- **Ownership enforced.** Only the owning user, or an `is_admin` user, may read
  a job. `is_admin` is the project's existing authorization convention.
- **Non-owners get 404, not 403.** Returning 403 would confirm the job id
  exists, letting any authenticated user enumerate job ids. This matches the
  app's "not found" convention.
- **Never exposed:** tracebacks, filesystem paths, driver errors, model
  internals, secrets, raw image bytes, Mongo `_id`.
  `test_status_response_never_leaks_internals` raises a job whose exception
  contains `/secret/path/cv2.so` and asserts the serialized response contains
  none of `/secret/path`, `Traceback`, `RuntimeError`, `cv2.so`, `.jpg`,
  `site-packages`, or the staged path.
- **Real exception logged server-side** with `log.error` + `traceback.format_exc()`.
- Uses existing `login_required` / `db_required`. No auth redesign.

---

## 7. Legacy `/process_plate` compatibility

`/process_plate` is **byte-identical to `HEAD`** — 6,512 chars, verified by
diffing the function body against `git show 3eec1777`. It is not redirected to
the async path. `process_license` … `/process_video`, camera and Socket.IO are
likewise unchanged.

`ealpr/utils.py`, `ealpr/ai_models.py`, `ealpr/blueprints/detection.py`,
`ealpr/image_storage.py`, `ealpr/image_thumbnails.py` are all verified
**UNCHANGED** — weights, thresholds, OCR preprocessing, Arabic handling,
confidence maths and `INFERENCE_LOCK` semantics are untouched. A test
(`test_ai_sources_are_byte_identical_to_head`) asserts this against the commit
so it cannot drift silently.

`test_10_process_plate_response_contract_is_unchanged` pins the 400 response
shape for a missing file and for a disallowed extension.

### Equivalence, proven with real inference

Same image, both paths, real YOLO/OCR, comparing the persisted
`DetectionResult`:

| Field | sync | async | match |
| --- | --- | --- | --- |
| `plate_number` | `ر٩` | `ر٩` | YES |
| `confidence` | 0.8987772464752197 | 0.8987772464752197 | YES |
| `ocr_confidence` | 0.7268192768096924 | 0.7268192768096924 | YES |
| `status` | unauthorized | unauthorized | YES |
| `visitor_name` | None | None | YES |
| `processed_by` | same user | same user | YES |

**EQUIVALENT: True.** Confidence values are bit-identical, confirming the
worker calls the same function with the same inputs.

---

## 8. Concurrency and `INFERENCE_LOCK`

`INFERENCE_LOCK` is **not** modified or bypassed. `run_job` calls the real
`process_license_plate`, which takes the lock via `predict_with_lock`. Model
concurrency is unchanged: still one inference at a time, one YOLO instance.

`test_14_inference_lock_is_still_the_serialisation_point` runs three jobs
concurrently against a fake that takes the *real* `INFERENCE_LOCK` and records
peak occupancy. Asserted peak is **1**.

A single consumer thread also bounds resource use: queue depth grows, threads
do not.

> **Not claimed:** real GPU inference concurrency was not tested. No throughput
> improvement is claimed. Correctness and controlled resource use were
> prioritised per C2.9.

---

## 9. Test results

| Suite | Result |
| --- | --- |
| `tests/test_inference_jobs.py` | **32 passed, 1 skipped** |
| Full discovery (A + B + C1 + C2) | **84 passed, 2 skipped, 0 failed (151.6s)** |
| Block A pagination | **PASS** |
| Block B live supplement | **58 passed, 0 failed** |
| Real-inference smoke (`EALPR_C2_SMOKE=1`) | **PASS** — 8.27s, plate `ر٩`, conf 0.8988 |

All 19 C2.10 cases are covered. Mocks/fixtures are used everywhere except the
single opt-in real-inference smoke test. Both skips are negative-control tests
that need data the live set does not contain.

### Two bugs found by the tests

- **Non-atomic job claim** (see §4) — would have produced duplicate
  `DetectionResult`s under concurrent runners.
- **Test-harness global leak.** My concurrency tests entered and exited
  `mock.patch` on `process_license_plate` from four threads at once. The final
  unpatch restored a stale value and **permanently leaked the fake inference
  function** into later tests — the smoke test silently reported
  `plate='__c2_test_…', confidence=0.87` instead of a real plate, and still
  "passed". Fixed by patching once in the main thread. Worth recording: a green
  smoke test was reporting fake data, and only comparing against the known real
  value exposed it.

---

## 10. Performance measurements

Real YOLO/OCR, real image (`20260917_171133_5ff0754d_original.jpg`, 31.3 KB):

| Measurement | Time |
| --- | --- |
| `POST /process_plate`, cold (incl. one-time model load) | 8.526s |
| `POST /process_plate`, **warm** (steady state) | 2.360s |
| **`POST /api/inference/jobs`** (202 returned) | **1.257s** |
| Worker: submit → terminal state (`completed`) | 3.892s |

**The POST does not wait for inference.** That is the required behaviour and it
is demonstrated, not assumed.

**Honest reading of the 1.9x.** The request-latency win is real but modest, and
smaller than it looks: the 1.257s POST is dominated by Atlas round-trips
(staging the file, inserting the job document), not by compute. The 8.5s→2.4s
cold→warm gap is a one-time model-load effect, not an async gain.

**The substantive benefit is isolation, not latency:** the request thread is
released immediately instead of being pinned for the inference duration, so
unrelated requests are no longer queued behind a running inference. That
enables concurrency — which I have deliberately **not** measured or claimed.

---

## 11. Data integrity

| Check | Before C2 | After C2 |
| --- | --- | --- |
| `detection_result` count | 66 | **66** |
| `visitor` count | 1023 | **1023** |
| Binary content digest | `42d3a96efc875ae0dda1d81a3e46927d` | **unchanged** |
| `authorized=True` / `False` | 812 / 211 | **812 / 211** |
| `status='authorized'` | 811 | **811** |
| Active (no `exit_time`) | 317 | **317** |
| Leftover `inference_job` docs | — | **0** |
| Leftover test users | — | **0** |
| Leftover `temp/` artifacts | — | **0** |

Every record created during testing was deleted **by id**; images and
thumbnails were removed by set difference against a pre-run snapshot — never a
time-based or pattern-based sweep. No visitor was modified: job plates use a
per-run `__c2_test_*` marker that matches no real visitor, so the
`entry_datetime_utc` side effect never fires against production data.

**One disclosure.** `user_activity` grew 381 → 394. The HTTP tests trigger the
application's pre-existing `after_request` hook, which appends a `UserActivity`
row per request. That is unchanged application behaviour, not something C2
introduced, but it means the suite is not literally zero-write database-wide.
No `visitor` or `detection_result` document was created, modified or deleted
outside the recorded, id-targeted test cleanup.

---

## 12. Files changed

| File | Change |
| --- | --- |
| `ealpr/inference_jobs.py` | **new** — queue, consumer thread, job lifecycle, failure handling |
| `ealpr/blueprints/inference.py` | **new** — the two endpoints, ownership, validation |
| `tests/test_inference_jobs.py` | **new** — 32 tests covering the 19 required cases |
| `models.py` | +60 lines, additive only — `InferenceJob` model |
| `ealpr/__init__.py` | +2 lines — register `inference_bp` |

Unchanged and verified byte-identical: `ealpr/utils.py`, `ealpr/ai_models.py`,
`ealpr/blueprints/detection.py`, `ealpr/image_storage.py`,
`ealpr/image_thumbnails.py`, `ealpr/repositories/*`, `config.py`, `app.py`.

No dependency was added. 43 → 45 registered routes.

---

## 13. Commit

```
phase2: add asynchronous inference job architecture
```

Tracked `__pycache__/*.pyc` files regenerated by the test runs are deliberately
left unstaged.

---

## 14. Remaining risks

1. **In-memory queue is lost on restart.** A job that is `queued` when the
   process dies stays `queued` forever. The document survives; the pending work
   does not. There is no reaper, and no client-visible timeout. **This is the
   most significant gap.**
2. **No watchdog for a hard-killed process.** A `processing` job survives in
   Mongo if the worker is killed mid-inference (`SIGKILL`, OOM, GPU fault). A
   client can poll such a job forever. Normal exceptions cannot cause this —
   `run_job` catches everything — but a hard process death can.
3. **No job retention policy.** `inference_job` grows without bound; no TTL,
   no pruning. Needs a cleanup job.
4. **Staged uploads leak if a job is abandoned while queued.** The `finally`
   cleanup only runs if the job actually executes. Jobs deleted from Mongo
   while still queued leave their file in `TEMP_DIR`.
5. **`_load_yolo_models` double-load race** (pre-existing, `ai_models.py:47`).
   Two threads can pass the unlocked `_models_loaded` check and both load
   models; one instance is discarded. C2 does not introduce this, but C2's
   worker thread makes a second caller marginally more likely. The POST
   endpoint's gate normally warms the models first, which reduces the window.
6. **The 1.26s POST is Atlas-latency bound.** Acceptable, but the async path
   is not "instant" — a client still waits ~1.3s for a 202.
7. **Dev server still in use.** Single-process, threaded, `debug=True`. Fine
   for a single inference at a time, but the in-process worker design depends
   on there being exactly one process. If this moves to gunicorn with multiple
   workers, **each process would run its own consumer thread and its own YOLO
   instance**, and `INFERENCE_LOCK` would stop being global. That migration
   would require a real external queue.

---

## 15. Recommended next block

Ordered by risk, not by convenience:

1. **Make jobs survive a restart.** A startup reaper that fails
   `queued`/`processing` jobs older than a threshold, so no job can hang
   forever. This closes risks 1 and 2 and is the single most valuable next
   step. It is also the honest prerequisite for trusting the 202 response in
   production.
2. **Bound storage.** TTL index or periodic pruning for `inference_job`, and
   clean up staged uploads for abandoned jobs (risks 3 and 4).
3. **Lock `_load_yolo_models`** with a module-level lock (risk 5).
4. **Fix the pre-existing `_video_jobs` store** to use the same durable
   `InferenceJob` model and the same ownership check. It currently loses all
   state on restart and lets any authenticated user read any video job. The
   reusable boundary for this now exists.
5. **Only then** consider migrating camera / video / Socket.IO onto the job
   boundary, per C2.7's deferral.

Deliberately **not** started: migrating `/process_plate` to async by default,
camera redesign, Socket.IO redesign, notifications, frontend polling, Celery
or Redis. `/process_plate` stays the compatibility path until a later stage
proves equivalence under production load.

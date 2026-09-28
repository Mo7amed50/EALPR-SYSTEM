# Phase 2 — Block C3 Report
## Asynchronous Inference Job Recovery, Cleanup and Retention Hardening

**Status:** Complete
**Baseline:** `a829096a` — *phase2: add asynchronous inference job architecture* (C2)
**Commit:** `phase2: harden asynchronous inference job lifecycle`
**Working directory:** `D:\EALPR SYSTEM`

---

## 1. Why this block exists

C2 made inference jobs durable by storing them in MongoDB while the work itself runs on an
in-process `queue.Queue` consumer. That left three ways for a job to hang forever, and none
of them required anything to go wrong beyond the process ending:

| # | Failure | Consequence in C2 |
|---|---------|-------------------|
| 1 | Process dies with jobs `queued` | Mongo documents survive, the in-memory queue does not. Nothing ever runs them; the client polls `queued` indefinitely. |
| 2 | Process dies with a job `processing` | The claim was already consumed. Even a manual requeue would not re-run it, and a retry risks a duplicate `DetectionResult`. |
| 3 | Process dies with a staged upload in `TEMP_DIR` | Cleanup only ran from `run_job`'s `finally`, so the file was never removed. |

C3 closes all three. It is deliberately conservative: **valid work survives, only work that
cannot be trusted is failed.**

---

## 2. What was built

### 2.1 Recovery policy (`recover_jobs`)

| Persisted state | Action | Rationale |
|---|---|---|
| `queued`, recent, owner exists | re-queue, run normally | legitimate work survives a restart |
| `queued`, recent, **owner deleted** | fail `startup_recovery` | `_persist_detection` stores `processed_by=job.user`; a dangling owner can never succeed, so running it would burn a full YOLO/OCR inference on a job with no possible outcome |
| `queued`, older than threshold | fail `stale_queued` + free file | past the trust boundary; holding its image is pure waste |
| `processing`, older than threshold | fail `stale_processing` + free file | **never retried** — the process may have died *after* persisting a detection and *before* completing the job, so a retry could duplicate it |
| `completed` / `failed` | untouched | already terminal |

A userless `queued` job is still runnable (`processed_by` is simply left empty); only a
**dangling** reference is unrunnable.

### 2.2 Thresholds — derived from C2 measurements, not round numbers

| Measurement (C2) | Duration |
|---|---|
| Cold `POST /process_plate` (one-time model load) | 8.526 s |
| Warm `POST /process_plate` (steady state) | 2.360 s |
| Async submit → terminal | 3.892 s |

* `PROCESSING_STALE_SECONDS = 600` — ~150× the measured warm inference and ~60× the
  measured end-to-end duration. Deliberately far too generous to kill a healthy job,
  including the first job after a restart which does pay the cold load. The requirement is
  *not to fail a healthy long-running job*, so erring generous is the correct direction.
* `QUEUED_STALE_SECONDS = 1800` — queue wait is not compute time and needs a much looser
  bound. At ~4 s per job, 30 minutes is a backlog of ~450 jobs; past that the job is not
  "waiting its turn", it is abandoned.

Both are env-overridable (`INFERENCE_QUEUED_STALE_SECONDS`, `INFERENCE_PROCESSING_STALE_SECONDS`)
so an operator can widen them for slow hardware without a code change. Widening is always
safe; a value that is too *small* produces a spurious failure, so these should only be
lowered with evidence about real inference duration.

### 2.3 Artifact cleanup — ownership-based, never blanket

Only files that can be **proven** to belong to an abandoned job are deleted:

1. **Name shape** — the exact regex `^inference_job_[0-9a-f]{32}_.+$` matches what
   `blueprints/inference.py` writes. Matching the whole shape, not just the prefix, is what
   makes it safe against a shared `TEMP_DIR`: the synchronous `/process_plate` and
   `/process_video` endpoints use a distinct `upload_` prefix and can never match.
2. **Ownership** — for the directory sweep, no `InferenceJob` document records the path as
   its `image_path`.
3. **Age** — applied only as a safety margin on top of the above, never as the sole
   criterion, because a file staged microseconds ago by a request that has not yet called
   `enqueue()` is briefly ownerless.

No directory is ever deleted, no blanket sweep is ever run, and a missing file is not an
error.

### 2.4 Retention (C3.6)

A TTL index on `completed_at` expires terminal jobs after 30 days
(`INFERENCE_JOB_RETENTION_SECONDS`, default `2592000`).

This works because **MongoDB's TTL monitor only acts on documents whose indexed field holds
a DATE**. A document whose `completed_at` is null or absent is never expired. Since
`completed_at` is written *only* by the terminal transitions, the database itself enforces
"expire completed + failed only" — the two states that must never be expired are structurally
safe rather than safe by convention.

Expiring a job does **not** touch its `DetectionResult`: the job only holds a `ReferenceField`
to the detection, and the collection defines no cascade that could reach it.

### 2.5 Model-init race (C3.7)

Before C3, `if _models_loaded: return` in `_load_yolo_models()` was unguarded, so two threads
calling a loader for the first time could both pass the check and both construct both models —
briefly doubling model memory and discarding one pair.

`_MODEL_LOAD_LOCK` now serialises *initialisation* only. This is **not** `INFERENCE_LOCK`:
`INFERENCE_LOCK` guards `model.predict` so CPU inference stays one-at-a-time, while the new
lock guards the once-per-process construction of the two YOLO objects. They solve different
problems and must not be merged — holding `INFERENCE_LOCK` across model loading would
serialise startup behind inference, and holding the new lock during `predict` would defeat
the non-freezing behaviour `INFERENCE_LOCK` exists to provide. The getters' outer check is a
lock-free fast path only; the authoritative decision is re-made under the lock.

### 2.6 One consumer, ever (C3.4)

* `_initialization_done` makes the startup hook idempotent.
* A structural `_all_workers` registry replaces trusting a single thread variable. A worker
  that outlives a `shutdown()` whose join timed out stays registered and is **reused**, never
  forgotten and then duplicated.
* `_queued_ids` tracks what is in the in-memory queue, so a forced second recovery cannot
  push the same id repeatedly. The atomic claim already made duplicate *execution*
  impossible; this is about not accumulating junk.
* `drain_queue()` keeps `_queued_ids` in step — otherwise a discarded id would stay marked as
  queued and recovery would permanently refuse to re-queue it.

### 2.7 Failure containment (C3.10)

`recover_jobs()` never raises: every step is guarded, one unusable job cannot stop the rest,
and a MongoDB outage degrades to "recovery skipped" rather than a failed boot. Recovery is
invoked from `create_app()` inside a `try/except` for the same reason. The status endpoint
returns fixed prose for the three recovery codes — never an exception string.

---

## 3. Two real bugs found and fixed during verification

### 3.1 The stale transition updated only ONE job per startup

`tests.c3_stale_count_check` showed 3 identical stale `queued` jobs producing
`stale_queued: 1`, with the other 2 left hanging — the exact failure this block exists to
prevent.

Cause: `QuerySet.modify()` is a `find_one_and_update` and therefore touches exactly one
document. That is *correct* for `_claim`, where exactly one runner must win, and *wrong* for
`_fail_stale`. With a single stale job the two are indistinguishable, which is why every
earlier test passed.

Fixed by using `update(multi=True)` for the bulk transition. The status filter is retained,
so a job claimed by a live worker mid-sweep is still spared, and the post-transition
read-back is the authoritative source for both the count and the files reclaimed.

Added `test_c3_5b` (3 stale queued) and `test_c3_7b` (2 stale processing) so the multi-document
case is covered explicitly.

### 3.2 C2's own test harness leaked 7 detections into live data

Block A and Block B started failing: detection count read 73 instead of 66, and
"exactly 6 path-backed records" read 13. The 7 extra records were `__c2_test_*` detections
left behind by **earlier C2 test runs** — the C2 harness did not record them for cleanup. Not
a C3 regression, but a genuine data-integrity defect in the C2 suite that would have kept
silently corrupting the production collection.

All 7 were confirmed as test artifacts (marker plate, `/tmp/` fake paths, no inline binary),
deleted by exact `_id` with a marker-guarded predicate, and Block A / Block B returned to
green. C3's harness now asserts the live detection count is unchanged in tearDown, so this
class of leak fails the suite instead of corrupting data.

---

## 4. Verification

All commands run from `D:\EALPR SYSTEM` against the live Atlas cluster and the real model
weights.

| Suite | Result |
|---|---|
| `tests/test_inference_jobs.py` (C2 + C3) | **79 tests, 78 passed, 1 skipped, 338.0 s** |
| C3-focused (recovery, processing, retention, cleanup, idempotency, model race) | **37 tests, 37 passed, 174.4 s** |
| C2-focused (API, lifecycle, contract, worker thread, data safety) | **32 tests, 32 passed** |
| Full suite (`unittest discover`) | **131 tests, 129 passed, 2 skipped, 384.1 s** |
| Real-inference smoke (`EALPR_C2_SMOKE=1`) | **PASS** — plate `ر٩`, confidence `0.8988`, 9.07 s |
| Block A regression (pagination) | **PASS** |
| Block B live supplement | **58 passed, 0 failed** |

The smoke result matches the C2 baseline exactly (`ر٩`, `0.8987772464752197`,
OCR confidence `0.7268192768096924`), confirming C3 changed no AI behaviour.

### 4.1 Performance

| Measurement | Result |
|---|---|
| Recovery, empty queue (paid on every start) | 0.846 s |
| Recovery, 25 recent + 5 stale + 40 terminal | 1.902 s — 25 requeued, 5 failed, 40 skipped, 0 errors |
| `ensure_indexes` cold / memoised | 1.104 s / 0.000002 s |
| Orphan artifact sweep | 0.221 s |
| 8-way concurrent first model access | 0.128 s, **2 constructions** (one pass = plate + OCR) |

The 2-construction figure is the model-race result: 8 threads racing for the first load
produce exactly one load pass, not eight. No throughput claim is made — Atlas round-trip
latency dominates these numbers.

### 4.2 Data integrity (before → after)

| Metric | Before | After |
|---|---|---|
| `DetectionResult` | 66 | **66** |
| `Visitor` | 1023 | **1023** |
| Binary content digest | `42d3a96efc875ae0dda1d81a3e46927d` | **identical** |
| Detections inline-binary / path-backed | 60 / 6 | **60 / 6** |
| Authorized | 812 | **812** |
| Unauthorized | 211 | **211** |
| Status = authorized | 811 | **811** |
| Active (no `exit_time`) | 317 | **317** |
| `InferenceJob` | 0 | **0** |
| `UserActivity` | 395 | 398 |

`UserActivity` grew by 3 purely from the pre-existing HTTP `after_request` logging fired by
the integrity script's own dashboard request. It is not written by C3.

---

## 5. Scope discipline

Changed — 5 files, exactly the intended surface:

* `ealpr/inference_jobs.py` — recovery, queue bookkeeping, artifact cleanup, retention init
* `ealpr/ai_models.py` — `_MODEL_LOAD_LOCK` only
* `models.py` — `JOB_RETENTION_SECONDS` and the TTL index
* `ealpr/__init__.py` — fail-soft startup hook
* `tests/test_inference_jobs.py` — 47 new C3 tests

Unchanged and verified: `ealpr/blueprints/inference.py` (POST + status API), `utils.py` and
`detection.py` (byte-identical), `process_license_plate`, `INFERENCE_LOCK`, camera/video,
Socket.IO, frontend, auth model, repositories, Redis/Celery/RQ, weights, thresholds,
preprocessing, confidence logic. A C2 regression test asserts the AI sources are byte-identical
and that `ai_models.py` differs by exactly the two documented C3.7 statements.

Tracked `__pycache__/*.pyc` files are modified by any test run and are deliberately **not**
included in the commit.

---

## 6. Residual risks

1. **Worker thread vanishing without a process restart.** Recovery runs at startup and on the
   next submit. A consumer thread that died *in place*, leaving a backlog and receiving no new
   submit, would not be noticed until restart. `run_job` catches `Exception` broadly, so this
   needs an uncatchable failure, and process death is the realistic case — which startup
   recovery does cover. A watchdog was considered and not added, to avoid a new thread and
   timer surface in a block explicitly not meant to change the worker architecture. This is
   the strongest argument for C4's supervisor.
2. **Transient Mongo outage at startup.** Recovery logs and returns; the app boots and new
   jobs run normally, but jobs already persisted at that moment are not re-queued until the
   next restart.
3. **Stale `processing` jobs are failed, not retried.** A user must resubmit. This is a
   deliberate correctness-over-convenience trade, not an oversight.
4. **Thresholds are hardware-dependent.** Correct for this machine's ~2.4 s warm inference. On
   markedly slower hardware the env vars must be widened — too small a value yields spurious
   failures, never silent data loss.
5. **30-day retention is an operator default.** It deletes job history, never detections.
   Lower it if audit requirements demand longer job retention.

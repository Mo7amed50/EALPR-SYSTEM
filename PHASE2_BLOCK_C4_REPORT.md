# PHASE 2 — BLOCK C4 REPORT

## Worker Supervisor & Runtime Reliability — IMPLEMENTATION

## 1. Status

**Complete.** C4 is implemented, tested and verified. The audit
(`PHASE2_BLOCK_C4_AUDIT_REPORT.md`) was completed first and is unchanged by this
work.

Two real defects found by the audit are closed, and the minimum supervision
needed to detect a missing consumer is now in place:

| Item | State |
|---|---|
| C4.1 Startup-stranded recent `processing` job | Fixed |
| C4.2 Startup MongoDB outage permanently disabling recovery | Fixed |
| C4.3 Duplicate-safety analysis | Confirmed, unchanged |
| C4.4 In-memory worker health fact | Added |
| C4.5 Supervision monitor | Added |
| C4.6 Restart through the one legal creation path, rate-limited | Added |
| C4.7 Lifecycle ordering on shutdown | Fixed |
| C4.8 No new thread-creation path | Proven by test |

Baseline for this block: `a5f98b6a` (C3). Final C3 performance and integrity
figures are unchanged and were re-measured as a control.

## 2. Baseline

Measured immediately before implementation:

| Metric | Value |
|---|---|
| `DetectionResult` | 66 |
| `Visitor` | 1023 |
| `InferenceJob` | 0 |
| Live users | 3 |
| Detections with inline binary | 60 |
| Path-backed detections | 6 |
| Binary content digest | `42d3a96efc875ae0dda1d81a3e46927d` |
| `authorized=True` / `False` | 812 / 211 |
| `status=authorized` | 811 |
| Active visitors (no `exit_time`) | 317 |
| Stale staged artifacts (pre-existing, ~20h) | 10 |

All ten stale artifacts pre-date this block. The read-only audit deliberately did
not delete them; they are ordinary leftovers of a previous crash, which is the
exact condition C4 exists to make self-healing.

## 3. Current Worker Architecture

Unchanged by C4, and deliberately so. The lazy-start design is preserved:

- `enqueue()` puts a job id in the in-memory `Queue` and mirrors it in
  `_queued_ids`; it does **not** start a consumer.
- `_ensure_worker(app)` is the only place a consumer thread is created, and it is
  idempotent under `_queue_lock`.
- An empty queue with no consumer is a valid, healthy state — not a fault.
- C4 adds a *second* thread, the monitor, which observes this state. It never
  consumes work; it only calls `_ensure_worker`.

```
enqueue() ──► Queue (in-memory, lost on restart) ──► _worker_loop ──► run_job
                    │                                              │
              _queued_ids mirror                              _claim()
                    │                                    (queued→processing,
                    ▼                                     atomic, one winner)
            worker_health()  ◄── monitor tick (5s) ──► _monitor_tick
                    │                                     └─► _ensure_worker
                    └── queue_depth, thread liveness, recovery_completed
```

## 4. Worker Lifecycle

| Thread | Name | Created in | Purpose | Stopped by |
|---|---|---|---|---|
| Consumer | `inference-worker` | `_ensure_worker` (only) | Drains the queue | `shutdown()` |
| Supervisor | `inference-monitor` | `_ensure_monitor` (only) | Ticks health every 5s | `shutdown()` via `_stop_monitor` |

`shutdown()` stops the **monitor first**. This ordering is the C4.7 fix: a live
monitor would immediately satisfy its own condition — work outstanding, no
consumer — and race the caller by restarting the thread that was just stopped.

Both threads log their start and exit. Those log lines carry only thread name,
state and counts: no plate text, no image paths (asserted by
`test_c4_24_worker_start_and_exit_are_logged`).

## 5. Failure Scenario Analysis

### 5.1 Unexpected Worker Exception
Unchanged (C3). `run_job` fails the job; the consumer survives. C4 adds nothing
that could alter this.

### 5.2 Silent Worker Death
**Now detected.** `worker_health()` reports `worker_running=False` from
`thread.is_alive()`. If `work_outstanding` is also true, the monitor calls
`_restart_worker` → `_ensure_worker`. Restart is rate-limited (§5.7).

### 5.3 MongoDB Outage
**Now recoverable.** See §7.2. Previously a startup outage disabled recovery for
the life of the process.

### 5.4 Inference Exception
Unchanged (C2). The job fails with `inference_error`; the consumer survives.

### 5.5 Application Restart
**Improved.** `recover_jobs(orphan_processing=True)` now also fails a
`processing` job of *any* age when no consumer is alive, closing the sub-600s
window (§7.1).

### 5.6 Multiple Consumers
Still impossible. C4 adds no creation path; `test_c4_23` asserts at source level
that exactly two `threading.Thread(` calls exist in the module — one in
`_ensure_worker`, one in `_ensure_monitor` — and that the monitor never names
itself as the consumer.

### 5.7 Worker Restart
Rate-limited by a sliding window, default **5 attempts per 300s**, configurable
via `INFERENCE_MAX_WORKER_RESTARTS` and
`INFERENCE_WORKER_RESTART_WINDOW_SECONDS`. On reaching the cap the monitor logs
once and suppresses further restarts until the window expires; outstanding work
then waits for a process restart. Without this, a worker that dies instantly
becomes an unbounded thread-churn loop.

## 6. Duplicate-Execution Risk

### The trace
`_claim()` atomically moves `queued → processing`. A `processing` job is no
longer in the in-memory queue.

### The dangerous window
`_persist_detection()` writes the `DetectionResult` and `_complete()` marks the
job completed. **These are two separate writes with no transaction between
them.** A crash between them leaves a detection with a non-terminal job.

### Does a worker restart trigger this window?
**No, and C4 preserves that.** `_claim()` matches only `status="queued"`, so a
restart cannot re-claim in-flight work, and a restart does not call
`recover_jobs` for a live consumer. `test_c4_10` asserts that repeated restarts
against a `processing` job leave both the status and the detection count
untouched, and `test_c4_11` asserts a `queued` job is re-run **exactly once**
across a restart — so a restart can neither duplicate work nor silently drop it.

### Visitor side effect
Unchanged. `_complete` re-uses the detection and only updates the visitor when
absent.

### Existing protection summary
| Protection | Status |
|---|---|
| Atomic `queued → processing` claim | Unchanged |
| `processing` never requeued | Unchanged, extended (§7.1) |
| Single consumer | Unchanged, re-proven |
| Restart cannot reach in-flight work | Unchanged, newly tested |

The residual crash window is **unchanged and explicitly not closed by C4**. It
is closed by making the user retry, never by automatic re-execution.

## 7. Current Recovery Interaction

### 7.1 C4.1 — startup-stranded recent `processing` job (the audit's main finding)

C3 recovered a `processing` job only once it was older than
`PROCESSING_STALE_SECONDS` (600s). A crash-and-immediately-restart — the most
common deployment event — lands *inside* that window, so recovery skipped the job
and it stayed non-terminal until some later restart at least 600s afterwards.

`recover_jobs` takes a new opt-in parameter:

```python
def recover_jobs(now=None, orphan_processing=False):
```

When it is true and `worker_is_running()` is false, **every** `processing` job
regardless of age is failed with `ERROR_STALE_PROCESSING` and its staged file is
reclaimed. It is never requeued.

Three deliberate design points:

- **It is never requeued.** Requeueing is the only action that could duplicate a
  `DetectionResult`.
- **It defaults to `False`**, so every C3 caller keeps exactly the C3 behaviour.
  Only startup recovery opts in, because only startup knows no consumer can be
  running.
- **It is guarded on `worker_is_running()`**, so it is a no-op whenever a live
  consumer exists. A forced re-run while a consumer is live cannot fail a job
  that is genuinely being worked on.

The count is reported under a new key `orphaned_processing`, kept separate from
the age-based `stale_processing` so the two causes remain distinguishable.

### 7.2 C4.2 — startup MongoDB outage permanently disabling recovery

C3 set `_initialization_done = True` *before* running recovery. If MongoDB was
unavailable at that moment, the guard stayed claimed for the life of the process:
jobs persisted before the outage could never be re-queued or failed, silently,
until someone restarted the process by hand.

The guard is now claimed **only on success**. If recovery reports errors, the
guard is left unset and the monitor retries via `_retry_recovery()` until MongoDB
answers. The first success claims the guard permanently. Retrying is safe because
recovery is idempotent. The monitor starts unconditionally for exactly this
reason — gating it on `counts["requeued"]` would disable supervision precisely
when it is needed.

### 7.3 Steady-state cost
None. `worker_health()` reads only the queue, the `_queued_ids` mirror and thread
liveness — no MongoDB round trip (`test_c4_8` asserts this by making any query
raise). Once recovery has succeeded, `_retry_recovery()` returns immediately
without touching the database, so the 5-second monitor tick performs **zero**
database work in steady state.

## 8. Current Test Coverage

26 new tests in `WorkerSupervisionTests`, all passing:

| Area | Tests |
|---|---|
| C4.1 orphan handling | `c4_1`, `c4_1b`, `c4_1c`, `c4_2`, `c4_3`, `c4_4` |
| Health fact | `c4_8`, `c4_9` |
| Supervision decision | `c4_5`, `c4_6`, `c4_7` |
| Restart duplicate-safety | `c4_10`, `c4_11` |
| Rate limiting | `c4_12`, `c4_13`, `c4_14` |
| Recovery retry | `c4_15`, `c4_16`, `c4_17`, `c4_18` |
| Monitor lifecycle | `c4_19`, `c4_20`, `c4_21`, `c4_22` |
| Structural guarantees | `c4_23`, `c4_24` |

The 79 C2/C3 tests are unchanged and still pass.

## 9. Missing Test Coverage

Stated honestly rather than papered over:

- **The monitor's real timer is not exercised end-to-end.** CPython cannot kill
  a thread, so "the worker dies" is modelled as the post-mortem *state* — work
  outstanding, no live consumer — which is the only state the supervision
  decision reads. This is stated in the test class docstring.
- `atexit`/signal-driven shutdown is not tested, because it was not implemented
  (§11.3).
- No test runs under gunicorn/systemd, so external restart behaviour is
  unverified.
- Multi-process deployment is unverified; the one-consumer guarantee is
  per-process, exactly as in C3.

## 10. Supervision Architecture Options

### Option A — In-Process Health Monitor
**Chosen.** What C4 implements: a daemon thread in `inference_jobs.py`.

### Option B — Application Supervisor
An OS-level supervisor (systemd/PM2) restarts the process. It cannot observe a
hung-but-alive worker, and it is a deployment change — out of scope.

### Option C — Deployment-Level Supervisor
As B. Not implemented.

### Comparison
Option A is the only option that can observe a *missing* consumer from inside
the process, needs no deployment change, and touches no public surface. Its
limitations are real and are recorded in §17.

## 11. Recommended C4 Implementation Boundary

### 11.1 Taken
- `recover_jobs(orphan_processing=...)` — opt-in, default off.
- `initialize_inference_jobs` claims its guard only on success.
- `worker_health()`, monitor thread, rate-limited restart via `_ensure_worker`.
- `shutdown()` stops the monitor first.
- Three new environment variables, all with defaults.

### 11.2 Deliberately not taken
- No public restart endpoint.
- No schema, TTL, or field changes.
- No change to `_claim`, `run_job`, `_persist_detection`, or `_complete`.

### 11.3 Explicitly out of scope
- **`atexit` and signal handlers were not added.** The audit listed them as
  optional. They were skipped: they are hard to test, can deadlock at interpreter
  shutdown, and buy little given `shutdown()` already exists for the test harness
  and is called on the paths that matter. This remains a residual risk (§17).
- No deployment, WSGI-server, or `debug` changes.

## 12. Files Expected to Change in Implementation

| File | Change |
|---|---|
| `ealpr/inference_jobs.py` | +354 / −6 — the whole implementation |
| `tests/test_inference_jobs.py` | +472 — 26 tests plus harness hardening |
| `PHASE2_BLOCK_C4_AUDIT_REPORT.md` | new (audit, unchanged) |
| `PHASE2_BLOCK_C4_REPORT.md` | new (this file) |

No other application file needed to change.

## 13. Files That Must Remain Untouched

All verified unmodified:

`models.py` · `ealpr/ai_models.py` · `ealpr/utils.py` ·
`ealpr/blueprints/inference.py` · `ealpr/blueprints/detection.py` ·
`ealpr/image_storage.py` · `ealpr/image_thumbnails.py` · `app.py` ·
`requirements.txt`

`test_c3_ai_and_detection_sources_are_still_byte_identical` continues to compare
four of these against `a829096a` and passes.

## 14. Data Integrity Assessment

| Check | Result |
|---|---|
| Detections 66 → 66 | Pass |
| Visitors 1023 → 1023 | Pass |
| Binary digest unchanged (`42d3a96e…`) | Pass |
| Binary/path split 60 / 6 | Pass |
| authorized 812 / 211, status 811, active 317 | Pass |
| Leaked `InferenceJob` documents | 0 |
| Leaked `__c2_test_` detections | 0 |
| Real-inference round trip restores its own record | Pass |

`user_activity` moved 398 → 402. That is the append-only dashboard login log
written by the verification scripts' own `/dashboard` requests, not a C4 effect.

## 15. Security Assessment

- No new endpoint, no new route, no new public surface.
- No new secret, credential, or dependency.
- No new query on any hot path; health is in-memory.
- Health logs contain thread names, booleans and counts only — no plate text, no
  image paths, no job payloads. Enforced by `test_c4_24`.
- The orphan rule can only ever move a job to `failed`; it grants no new access to
  job data.

## 16. Performance Considerations

- Monitor overhead: one daemon thread, one wakeup per 5s, a handful of
  `is_alive()` checks and a `qsize()`. No I/O.
- Recovery retry: at most one `recover_jobs()` attempt per tick, and only while
  recovery is still incomplete. This is self-limiting rather than a busy loop.
- Startup: one extra guarded `processing` query, once, only when
  `orphan_processing=True`.
- C3's measured performance figures (warm inference ~2.4s, full 105-test module
  ~456s) remain in the same range; the C4 module run was 130s for 26 tests.

## 17. Residual Risks

1. **The `_persist_detection` → `_complete` crash window is unchanged.** C4
   cannot reach it and does not pretend to.
2. **No `atexit`/signal hook** (§11.3). A hard kill still relies on startup
   recovery — which C4.1 now makes effective within seconds rather than 600s.
3. **Restart cap means a worker that dies instantly stops being restarted**;
   outstanding work waits for a process restart. This is intentional, and the
   log line says so.
4. **Fixed 5s tick.** Not adaptive; a long Mongo outage means one retry attempt
   per tick.
5. **Per-process guarantee.** Multi-worker deployments still get one consumer per
   process, unchanged from C3.
6. **`debug=True` with the Werkzeug development server** remains the runtime
   entry point. Out of scope for C4 and worth addressing in a later block.

## 18. Proposed C4 Implementation Plan

1. Read the audit; fix the `recover_jobs` signature and the `initialize_inference_jobs`
   guard ordering. — **Done**
2. Add `worker_health()`, the monitor, and rate-limited restart through
   `_ensure_worker`. — **Done**
3. Stop the monitor first in `shutdown()`. — **Done**
4. Write the 26 tests. — **Done**
5. Regress everything, write this report, commit once. — **Done**

## 19. Verification Plan

| Check | Result |
|---|---|
| C4 test class (26) | **OK** |
| `tests.test_inference_jobs` (105 = 79 + 26) | **OK**, 1 skipped, 455.8s |
| Full suite (157) | **OK**, 2 skipped, 524.3s |
| Block A pagination | **PASS** — incl. `summary shows 21-40 of 66` |
| Block B live supplement | **58 passed, 0 failed** |
| Real-inference smoke | **PASS** — plate `ر٩`, confidence `0.8988`, HTTP 200 in 27.2s |
| Post-run integrity | **66 / 1023 / digest `42d3a96efc875ae0dda1d81a3e46927d`** |
| Job/detection leak audit | **0 / 0**, 3 live users |

### Two real defects found during implementation

**1. A production-code bug in the new rate limiter.** `_consume_restart_budget`
assigned `_restart_suppressed` and `_restarts_total` without declaring them
`global`, so Python treated them as function locals and the *first* supervision
restart raised `UnboundLocalError` instead of restarting anything. Found by
`test_c4_12`/`c4_13`; fixed by declaring both globals.

**2. A test-harness leak that corrupted live data.** Before C4, a job only ever
finished on the thread its own test started, so recording the job id at creation
was enough. C4's supervision can finish a job on the test's behalf, and
`test_c4_7` leaked one `DetectionResult` into the production collection. It did
not fail the suite; it surfaced one run later as a Block A failure
(`67 != 66`), which is exactly the failure mode the C3 report describes. The
document was identified by its `__c2_test_` marker and removed with the existing
marker-scoped cleanup script, restoring 66. The harness was then hardened so the
class of leak cannot recur:

- `_cleanup` now deletes any `__c2_test_` detection by marker, so a completion
  landing after a test recorded its ids is cleaned instead of persisting.
- `_quiesce_worker` now asserts the **monitor** stopped too, not just the
  consumer — a surviving supervisor can start consumers on its own schedule and
  pick up the next test's queued job under half-applied mocks.

Deletion is strictly marker-scoped, so it can never touch a real record.

## 20. Final Verification Verdict

C4 is complete. The audit's two concrete defects are fixed and covered by tests;
the duplicate-execution risk analysis is unchanged and now has explicit tests
pinning it. Full suite, both regression suites, the real-inference round trip and
the data-integrity check all pass, and the database is byte-identical to the
pre-C4 baseline apart from the append-only dashboard login log.

**Block C5 has not been started.**

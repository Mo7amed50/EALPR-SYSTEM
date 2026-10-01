# Phase 2 — Block C5 (S1, S3, S4): Runtime Mode, Single-Process Policy, Exit Shutdown

**Date:** 2026-10-02
**Baseline commit:** `c1814880` (`phase2: harden worker supervision and recovery`)
**Scope:** three items from the C5 audit (`PHASE2_BLOCK_C5_AUDIT_REPORT.md`) — S1, S3, S4 only.
**Status:** implemented, verified, committed.

---

## 1. Scope

Three audit items were requested and only these three were implemented.

| Item | Audit finding | Delivered |
|------|---------------|-----------|
| **S1** | R1, R2, R12 — `debug=True` hard-coded in the entry point | Environment-driven `EALPR_ENV` / `EALPR_DEBUG`, validated at startup |
| **S3** | R4, R6, R7, R11 — multi-process is *incorrect*, not merely untested, and nothing enforced it | `config.assert_single_process()` called from the factory; rejects a second process with a clear error |
| **S4** | R5, V11 — `shutdown()` existed and was correct, but only tests ever called it | One idempotent `atexit` registration calling that existing function |

**Deliberately not implemented** (out of scope by instruction):

- **S2** — production WSGI server selection. Requires an owner decision; no server installed.
- **S5, S6, S7** — secret/startup validation, logging discipline, dependency and deployment documentation.

---

## 2. Files changed

| File | Change |
|------|--------|
| `config.py` | Added the C5 runtime block: `VALID_ENVIRONMENTS`, `RuntimeConfigError`, `_env_flag()`, `resolve_runtime_mode()`, `assert_single_process()` |
| `ealpr/__init__.py` | `create_app()` resolves and enforces the runtime mode before building Flask; registers the S4 hook next to `initialize_inference_jobs()` |
| `ealpr/inference_jobs.py` | Added `import atexit`, `register_atexit_shutdown()`, `_atexit_shutdown()`, `atexit_shutdown_registered()` |
| `app.py` | `debug=True` replaced with the validated value; `use_reloader=False` kept explicit with a comment recording why |
| `tests/test_runtime_lifecycle.py` | **New.** 25 focused tests for S1/S3/S4 |

Nothing else was modified. No dependency changes, no schema changes, no UI changes.

---

## 3. S1 — Environment-driven runtime mode

### Configuration

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `EALPR_ENV` | `development`, `production`, `test` | `development` | Selects the runtime mode |
| `EALPR_DEBUG` | `1/true/yes/on`, `0/false/no/off` | `0` (off) | Enables the interactive debugger — **development only** |

`EALPR_ENV` is normalised for surrounding whitespace and case only. Every other
value is a configuration error.

### Behaviour

| Mode | `EALPR_DEBUG=0` (default) | `EALPR_DEBUG=1` |
|------|--------------------------|-----------------|
| `development` | Boots, no debugger | Boots, debugger on |
| `production` | Boots, no debugger | **Refuses to start** |
| `test` | Boots, `TESTING=True` | **Refuses to start** |

### Design decisions

1. **Debug requires development, everywhere else is refused.** The audit
   specified the production case (RCE path). The same reasoning applies to
   `test`: the Werkzeug debugger intercepts the exception the suite needs to
   report, turning a failure into a hang. Implementing one rule
   (`debug and env != 'development'`) avoids two similar rules drifting apart.

2. **An invalid combination raises rather than falling back.** S1 requires that
   an invalid production configuration must not be silently downgraded. A
   `RuntimeConfigError` stops startup instead of running with settings the
   operator did not ask for.

3. **Checks run before the `Flask` object exists.** A rejected configuration
   fails cleanly rather than leaving a half-built app behind.

4. **The default is development with debug off.** This preserves developer
   usability (`python app.py` still serves the app) while removing the
   accidental-debugger risk that the hard-coded `debug=True` created.

5. **The entry point no longer decides anything.** `app.py` reads the value the
   factory already validated (`app.config["EALPR_DEBUG"]`). There is no second
   place where `debug` can be set.

---

## 4. S2 — Production server: NOT IMPLEMENTED

Explicitly excluded by instruction. `app.py` still uses `socketio.run()` with
`allow_unsafe_werkzeug=True`, which is Werkzeug's documented development
behaviour. `host="0.0.0.0"` and `port=5000` are unchanged.

Turning off debug while still on the development server is the safe
intermediate state the audit recommended; introducing a production server while
`debug=True` was the unsafe ordering, and that ordering has now been removed.

---

## 5. S3 — Single-process enforcement

### The constraint

The C2/C3/C4 design is process-local by construction:

| Component | Why a second process breaks it |
|-----------|-------------------------------|
| In-memory job queue | Each process would have its own queue; `queued → processing` bookkeeping would not be shared |
| One-consumer guarantee | `_claim()` in Mongo prevents duplicate *inference*, but not duplicate *consumers* — C4's restart budget and self-supervision are per-process |
| C4.1 orphan recovery | `recover_jobs(orphan_processing=…))` assumes the only other process was dead; with a live second process it would re-queue a job that is genuinely running |
| Live Socket.IO broadcasts | `async_mode="threading"` with no `message_queue`: an event emitted in process A never reaches clients attached to process B |

The correct response is to enforce the constraint, not to change C4.

### Enforcement

`config.assert_single_process()`, called from `create_app()` — the single place
every entry path (`app.py`, a future WSGI runner, a test) passes through —
rejects the two settings that actually create extra processes:

| Variable | Rejected when | Rationale |
|----------|---------------|-----------|
| `EALPR_RELOADER` | on | **The near-miss.** R4 is exactly this: a one-token edit to `use_reloader=False` silently creates a dual-process deployment, and it is invisible once enabled |
| `EALPR_WORKERS` | anything but `1` | Makes the constraint an explicit, declared value rather than a comment |

No PID scanning, no lock files, no broker. The check is a two-variable
configuration guard.

`EALPR_RELOADER` defaults to off, so the development experience is unchanged.

---

## 6. S4 — Interpreter-exit shutdown

### The problem

`shutdown()` existed and was already correct, but `git grep` found no
application caller — only tests. A normal interpreter exit therefore left the
consumer thread and the C4 monitor to be killed by the runtime. An in-memory job
was dropped silently: its MongoDB row stays `processing` and only the C3/C4
recovery pass on the *next* start notices.

### The change

`register_atexit_shutdown()` registers `_atexit_shutdown`, which calls the
existing `shutdown()`. There is no new shutdown logic. Called from `create_app()`
next to `initialize_inference_jobs()`.

Requirements met:

- **One registration.** Guarded by a module-level `_atexit_registered` flag;
  a second `create_app()` returns `False` and registers nothing.
- **No data mutation.** `shutdown()` only touches threads and the in-memory
  queue; `drain_queue()` discards in-memory ids only. Verified against the live
  database (see §8).
- **No new requeue or persistence.** Untouched.
- **Never raises.** The callback catches everything. `atexit` runs during
  interpreter teardown, when logging and threading are partially torn down; a
  raise there would mask the process's real exit status.
- **Safe when repeated.** `shutdown()` is already idempotent (C4.21); verified.

---

## 7. Tests added

`tests/test_runtime_lifecycle.py` — **25 tests, all passing.** Read-only with
respect to the database; no worker is started by the suite as a side effect.

| Group | Tests | Covers |
|-------|-------|--------|
| `RuntimeModeTests` | 12 | Default mode; development+debug; production; **production+debug refused**; `test` determinism; `test`+debug refused; invalid env; invalid boolean; whitespace/case tolerance; all documented modes |
| `SingleProcessTests` | 7 | Default accepted; `EALPR_WORKERS=1` accepted; **`>1` refused**; **reloader refused**; reloader off accepted; non-integer refused; production does not relax the rule |
| `AppFactoryRuntimeTests` | 2 | Factory records the mode; factory refuses invalid debug and multi-process configurations |
| `AtexitShutdownTests` | 4 | Registration is idempotent; callback invokes the real `shutdown()`; callback never raises; **shutdown does not mutate `detection_result` or `visitor`**; repeated calls are safe |

Each test runs inside a `runtime_env()` context manager that clears all four
variables and restores them afterwards, so no value can leak between cases and
the developer's own shell or `.env` cannot change an expected result.

---

## 8. Data integrity

Baseline taken **before** any edit and re-taken **after** all verification.

| Metric | Pre | Post | Match |
|--------|-----|------|-------|
| `detection_result` count | 66 | 66 | ✅ |
| `visitor` count | 1023 | 1023 | ✅ |
| `inference_job` count | 0 | 0 | ✅ |
| `original_image` bytes | 6,355,490 | 6,355,490 | ✅ |
| `processed_image` bytes | 946,803 | 946,803 | ✅ |
| path-backed detections | 6 | 6 | ✅ |
| content digest | `9200c7588833bb5456221036de1d6913` | `9200c7588833bb5456221036de1d6913` | ✅ |
| authorized / unauthorized | 812 / 211 | 812 / 211 | ✅ |
| active visitors | 317 | 317 | ✅ |

### A note on the C4 digest

The C4 digest and this one **do not match, and that is expected.** The C4 figure
(`42d3a96efc875ae0dda1d81a3e46927d`) was produced by hashing documents in
MongoDB **natural order**. That digest is order-dependent, and the natural order
is not stable across delete/insert cycles:

- Natural order differs from `_id`-sorted order at **34 of 66 positions**.
- Recomputing the identical content `_id`-sorted gives `9200c758…` consistently.
- Per-field byte totals match C4 **exactly** (6,355,490 original — C4's Block B
  measured `original_image` only; combined with `processed_image` that is
  7,302,293).

Investigation found **0** documents with a `__c2_test_` plate marker and **0**
detections newer than 3 days, so no test artefact and no new production record is
present. Content is intact; the C4 digest was simply an unreliable signal.

C5 therefore uses an `_id`-sorted digest as the authoritative baseline.

---

## 9. Regression results

| Suite | Baseline (pre-change) | Post-change | Result |
|-------|----------------------|-------------|--------|
| `tests.test_inference_jobs` (C2/C3/C4) | 105 OK, 1 skipped | 105 OK, 1 skipped | ✅ no regression |
| Full suite (`unittest discover`) | 157 OK, 2 skipped | **182 OK, 2 skipped** | ✅ +25 new, no regression |

Runtime: inference suite 485.8s, full suite 536.1s.

### End-to-end verification

Run as real subprocesses through the actual entry point (`import app`, which
executes `create_app()` at import time):

| Scenario | Result |
|----------|--------|
| No variables set | Boots `development`, `debug=False`, hook registered |
| `EALPR_ENV=development EALPR_DEBUG=1` | Boots `development`, `debug=True` |
| `EALPR_ENV=production` | Boots `production`, `debug=False` |
| `EALPR_ENV=production EALPR_DEBUG=1` | **Refused** with a clear message |
| `EALPR_ENV=test` | Boots `test`, `debug=False` |
| `EALPR_RELOADER=1` | **Refused** |
| `EALPR_WORKERS=4` | **Refused** |
| `EALPR_WORKERS=1` | Boots normally |

### S4 control experiment

A live consumer thread was started in a child process, and an observer was
registered *before* `create_app()` so that, with atexit's reverse ordering, it
ran *after* the C5 hook:

| | Worker running | Monitor running | Shutdown event set |
|---|---|---|---|
| **C5 hook active** | `false` | `false` | `true` |
| **Control (hook disabled)** | `true` | `true` | `false` |

The difference is attributable to the C5 hook alone.

---

## 10. Preservation rules honoured

Unchanged, as required:

- `_claim()` — atomic `queued → processing` transition
- `run_job()` and the inference pipeline
- `_persist_detection()`, `_complete()`, job lifecycle semantics
- Queue behaviour (`enqueue`, `drain_queue`, `_queued_ids`)
- `InferenceJob` schema — no new fields
- The C4 worker-supervision algorithm — monitor thresholds, restart budget, self-supervision, orphan rules
- YOLO/OCR models, camera, video, UI, templates, static, authentication, repositories, image storage
- Socket.IO behaviour (`threading`, no `message_queue`); no broker or distributed lock added
- Dependencies — `requirements.txt` untouched; no public endpoint added

---

## 11. Remaining C5 items

| Item | Description | Status |
|------|-------------|--------|
| S2 | Production WSGI server entry point | **Not implemented** — needs an owner decision |
| S5 | Refuse production startup with default `SECRET_KEY` / admin password | Not implemented |
| S6 | Replace `print()` in the startup path with the application logger | Not implemented |
| S7 | Dependency and deployment documentation | Not implemented |

Additional follow-ups noted during implementation:

- `.env.example` does not yet document the four new variables. It was left
  untouched to keep this block inside its stated scope; the variables are
  documented in `config.py` and in this report.
- `host="0.0.0.0"` remains in `app.py`. It is defensible with debug off (no
  debugger on the exposed interface) but is a deployment decision.
- `_connect_mongodb` and `_create_default_admin` still `print()` (this is S6).

---

## 12. Risk and rollback

**Residual risk introduced by this block:** very low. The switches default to
the safe side (`development`, debug off, one process), so an operator who sets
nothing gets strictly safer behaviour than before — the debugger is no longer
enabled by default. The main operational effect is that `python app.py` no
longer enables the debugger unless `EALPR_DEBUG=1` is set.

**Rollback:** revert the single commit. No schema, data, or dependency change
is involved, so rollback is safe at any time.

---

## 13. Conclusion

S1, S3 and S4 are implemented, verified end-to-end, and free of regressions.
The full suite is green at 182 tests (25 new), and live data is byte-identical
to the pre-change baseline. S2 remains outstanding and requires an owner
decision on production server selection.
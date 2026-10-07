# Phase 2 — Block C5 (S5, S6, S7)

**Secrets, startup logging, and runtime/deployment documentation**

| | |
|---|---|
| Base commit | `ca51de4f` — *phase2: harden runtime mode and shutdown lifecycle* |
| Deliverables | S5 production secret validation, S6 startup logging discipline, S7 documentation |
| Full suite | **210 passed, 2 skipped** (baseline 182 passed, 2 skipped + 28 new) |
| Data integrity | Unchanged — digest `9200c7588833bb5456221036de1d6913` before and after |
| Commit | `phase2: harden production secrets logging and docs` (exactly one) |

---

## 1. Scope

Block C5 S5, S6 and S7 were implemented. S1 (runtime mode validation), S3
(single-process enforcement) and S4 (shutdown lifecycle) were completed and
committed in `ca51de4f` and were **not** reopened.

**Delivered in this commit**

- **S5** — production startup refuses a missing or placeholder `SECRET_KEY` and
  a missing or default `DEFAULT_ADMIN_PASSWORD`.
- **S6** — the startup path reports through the `logging` module instead of
  `print()`, with credential redaction.
- **S7** — `.env.example` and `README.md` document the four runtime variables,
  the runtime modes, the production requirements, the single-process rule, the
  debug restriction, the startup command, and the fact that S2 is not
  implemented.

**Explicitly out of scope** — see §10.

---

## 2. S5 — Production secret and admin credential validation

### Why the check exists

`SECRET_KEY` and `DEFAULT_ADMIN_PASSWORD` both had development fallbacks
(`ealpr-dev-secret` and `admin123`). A fresh clone boots with no `.env` at all,
which is the correct local behaviour and a serious production failure: a shared
Flask `SECRET_KEY` means anyone can forge session cookies, and a known admin
password means the login form is decoration.

### Design

`config.assert_production_secrets(env_name, secret_key=None, admin_password=None)`
raises the existing `RuntimeConfigError` (the same exception type S1 and S3
already use, so there is one startup-rejection mechanism, not three).

- Runs in `create_app()` **before the Flask object exists**, immediately after
  `assert_single_process()`, alongside the S1 and S3 checks. A rejected
  configuration therefore stops startup instead of producing a running service
  that silently accepts default credentials.
- Validates the exact values the factory is about to configure — the factory
  passes `SECRET_KEY` and `DEFAULT_ADMIN_PASSWORD` explicitly rather than
  re-reading the environment, so what is checked is what would be used.
- Non-production modes return immediately and keep their fallbacks. This is what
  preserves the zero-setup local workflow and keeps the test suite deterministic
  with no real secrets.
- Every problem found is collected and reported in **one** message, so an
  operator fixes both variables in a single round trip rather than one per boot
  attempt.

### What is refused in production

| Condition | Refused |
|---|---|
| `SECRET_KEY` missing / blank / whitespace | yes |
| `SECRET_KEY` = `ealpr-dev-secret` (development fallback) | yes |
| `SECRET_KEY` = `your-super-secret-key-change-this` (`.env.example` template) | yes |
| `DEFAULT_ADMIN_PASSWORD` missing / blank | yes |
| `DEFAULT_ADMIN_PASSWORD` = `admin123` (known default) | yes |
| Any other non-empty value | accepted |

The two known-bad values are now named constants —
`config.DEVELOPMENT_SECRET_FALLBACK`, `config.KNOWN_DEFAULT_ADMIN_PASSWORD` and
the `config.PLACEHOLDER_SECRET_KEYS` frozenset — rather than inline string
literals, so the validator and the defaults cannot drift apart silently.

### Values are never echoed

The error message names the **variable** and describes the **kind** of problem.
It never interpolates the rejected value. This matters beyond politeness: these
`RuntimeConfigError` strings end up in crash output, CI logs and terminal
scrollback, which is exactly where a credential should not appear.

Example, captured from a real subprocess run:

```
Production startup refused. This configuration problem must be fixed before
EALPR_ENV=production will run: SECRET_KEY is a built-in development placeholder
rather than a unique random value. Set real values in the environment (see
.env.example). The values themselves are never echoed here.
```

### Rejection is verified at the real entry path

Seven subprocess cases were run through `python` → `create_app()`, not just
in-process unit tests:

| Case | Configuration | Result |
|---|---|---|
| A | production + valid `SECRET_KEY` + valid admin password | boots, `env=production debug=False` |
| B | production + `ealpr-dev-secret` | refused; value not echoed |
| C | production + `your-super-secret-key-change-this` | refused; value not echoed |
| D | production + `admin123` | refused; value not echoed |
| E | production + both empty | refused; **both** variables named in one message |
| F | development, no secrets at all | boots |
| G | test, no secrets at all | boots |

---

## 3. S6 — Startup logging discipline

### What was wrong

The startup path reported through `print()`, so output went to stdout with no
levels, no timestamps and no way to suppress or redirect it. Worse, nothing in
the project configured logging at all: `logging.getLogger(__name__)` calls in
`ealpr/inference_jobs.py` were emitting records that nobody collected, so every
INFO and WARNING the C2/C3/C4 worker code already produced was silently
discarded.

### What changed

**`ealpr/__init__.py`**

- `_ensure_startup_logging()` attaches **one** `StreamHandler` to the `ealpr`
  logger at `INFO`, formatted `%(levelname)s: %(name)s: %(message)s`. It is
  idempotent (`if log.handlers: return`) so repeated `create_app()` calls cannot
  stack handlers. The handler is scoped to the `ealpr` logger only — the root
  logger is untouched, so third-party loggers keep their own levels. This is not
  a global logging redesign.
- `logging.basicConfig()` was deliberately avoided: it would reconfigure the
  root logger and therefore every dependency's output as well.
- `_ensure_startup_logging()` is the **first** thing `create_app()` does, so the
  rejection messages from S1/S3/S5 are reported through the logger too.
- Ten `print()` calls replaced with logger calls at mapped levels:

| Startup event | Level |
|---|---|
| `Starting EALPR in %s mode` | INFO |
| `Connecting to MongoDB (attempt n/m)` | INFO |
| `MongoDB connected to database %s` | INFO |
| `Created default admin user.` | INFO |
| No MongoDB URI configured (read-only fallback) | WARNING |
| MongoDB URI missing a scheme (read-only fallback) | WARNING |
| MongoDB attempt failed | WARNING |
| All MongoDB connection attempts failed | ERROR |
| Inference job recovery could not start | WARNING |
| `DEFAULT_ADMIN_PASSWORD` still the development default | WARNING |
| Could not create default admin | WARNING |

Levels were chosen by meaning, not by habit: **INFO** is normal progress,
**WARNING** is a degraded-but-running state, **ERROR** is a failure.

**`config.py`**

- Two model-path warnings converted to `log.warning(...)`. `config` cannot use
  `app.logger` because it is imported before any Flask object exists, so it uses
  its own module logger. At `WARNING` or above the stdlib fallback handler emits
  the record even with no handler configured, which keeps these two visible
  without configuring logging globally.

### Credential redaction

`_redact_credentials(text)` strips `user:password@` (replacing it with `***@`)
and is applied on the way out to every startup message that carries an
operator-supplied string or an exception. Host names, database names and paths
are retained — they are operationally useful; only credentials are removed.

Applied to:

- MongoDB attempt failure text (drivers routinely quote the URI they were given).
- `could not create default admin` and `inference job recovery could not start`.
- **Not** needed for the invalid-URI message, because that message no longer
  includes the URI at all. The original code interpolated the whole raw value
  (`Invalid MongoDB URI format: {uri}`), which was a live credential leak — the
  string had already failed a scheme check, so it could be a mistyped
  `user:password@host` rather than a URL. It is now a scheme-only message with no
  fragment of the input.

### Intentionally left unchanged (prints outside S6's scope)

S6 covers the startup path. These `print()` calls are runtime or lazy-load
diagnostics and were left alone deliberately:

- `ealpr/ai_models.py` L63/69/73/75 — lazy YOLO model load progress (out of
  scope: no AI/OCR changes permitted).
- `ealpr/inference_jobs.py` L726/728 — runtime worker diagnostics.
- `ealpr/utils.py`, blueprints, `scripts/*` — unrelated to startup.

No test in the repository asserts on startup `print()` text (verified by grep),
so no test expectations were rewritten.

---

## 4. S7 — Runtime and deployment documentation

### `.env.example`

Still ships placeholders only. Added:

- A comment on `SECRET_KEY` explaining that the shipped value is a development
  placeholder refused in production, plus the `secrets.token_hex(32)` command to
  generate a real one.
- An empty `DEFAULT_ADMIN_PASSWORD=` line, documented as required in production.
- A full **Runtime mode** section documenting `EALPR_ENV`, `EALPR_DEBUG`,
  `EALPR_RELOADER` and `EALPR_WORKERS` with their defaults, allowed values and
  failure behaviour.
- `EALPR_WORKERS=1`, `EALPR_RELOADER=false`, `EALPR_DEBUG=false`,
  `EALPR_ENV=development` as concrete commented-out entries.
- A note that the startup command is `python app.py` and that **Block C5 S2 is
  NOT implemented**.

### `README.md`

Added a *Runtime Modes and Production Readiness (Phase 2, Block C5)* section
covering, in order:

1. The startup command: `python app.py`.
2. All three runtime modes (`development`, `production`, `test`) and when each
   applies, including that validation happens before any Flask object or DB
   connection exists.
3. All four runtime variables with defaults and failure behaviour, including
   that `EALPR_DEBUG` is only honoured under `EALPR_ENV=development` and raises
   `RuntimeConfigError` otherwise.
4. The single-process requirement, stated as a **correctness** requirement (one
   queue consumer, one orphan-recovery owner, one in-process Socket.IO registry)
   rather than a preference, plus the explicit `EALPR_WORKERS=1` /
   `EALPR_RELOADER=false` settings.
5. The Socket.IO architecture: threaded within that single process via
   flask-socketio, which is precisely why adding processes would require an
   external message queue the application does not have.
6. Deployment status: **S2 is not yet implemented**, no production WSGI server is
   installed or configured, and the application is **not production-ready**.
7. A statement that `.env.example` values are placeholders and real credentials
   must not be committed.

The section does not claim production readiness anywhere, and no production
credential appears in either file.

---

## 5. Tests

New file: `tests/test_secret_logging_docs.py` — **28 tests**, three classes.

**`ProductionSecretValidationTests` (13)**

- production boots with valid secrets (and reports `production` / `debug=False`)
- production refuses missing `SECRET_KEY`
- production refuses the development fallback secret (and does not echo it)
- production refuses the `.env.example` placeholder secret (and does not echo it)
- production refuses a missing admin password
- production refuses the known default admin password (and does not echo it)
- error message never echoes values — valid values are not echoed when a
  *different* field fails, and refused values are not echoed even when several
  problems are reported together
- multiple problems are reported together in one message
- development still accepts the fallbacks (zero-setup guard)
- test mode is deterministic and needs no secrets (two runs, identical result)
- `create_app()` refuses production with default secrets (boot-level)
- `create_app()` is unaffected outside production

**`StartupLoggingTests` (7)**

- startup sources use logging, not `print` — regex `print\s*\(` must not appear
  in `config.py` or `ealpr/__init__.py`
- `create_app()` emits INFO logs and the mode banner is present
- startup logs contain no sensitive values (canary `SECRET_KEY` and
  `DEFAULT_ADMIN_PASSWORD` sentinels must not appear in captured logs)
- `_redact_credentials` strips userinfo but keeps host and database name
- `_redact_credentials` works on exception text
- an invalid MongoDB URI is never logged — a URI containing a sentinel password
  is passed through the real `_connect_mongodb()` path; the log must not contain
  it while still reporting the failure
- the startup logging handler is installed at most once (idempotency)

**`DocumentationTests` (8)**

- both documents contain all four runtime variables
- README states the single-process requirement
- README states the debug restriction (`EALPR_ENV=development`, `RuntimeConfigError`)
- both documents state the production secret requirements, naming the two
  refused values
- README states the startup command
- README states that S2 is not yet implemented
- no production-readiness claim is made
- no WSGI server is documented as available in either file (guards against S2
  being accidentally claimed or partially implemented)
- `.env.example` ships only placeholders (`SECRET_KEY` is a known placeholder,
  `DEFAULT_ADMIN_PASSWORD` is empty, `MONGODB_URI` contains `<`)

No test needs a real secret, password or connection string, so the suite remains
deterministic on a clean checkout.

---

## 6. Verification results

| Suite | Result |
|---|---|
| `tests.test_secret_logging_docs` (new, 28) | **OK** — 28/28 |
| `tests.test_runtime_lifecycle` (C5 S1/S3/S4, 25) | **OK** — 25/25 |
| `tests.test_inference_jobs` (C2/C3/C4, 105) | **OK** — 105 passed, 1 skipped |
| Full suite before new tests (baseline check) | **OK** — 182 passed, 2 skipped |
| **Full suite with new tests (final)** | **OK** — **210 passed, 2 skipped** |
| Subprocess e2e (7 production/dev/test cases) | **ALL PASSED** |

The full suite was run twice: once with the S5/S6 code in place and no new
tests, confirming the baseline of 182 passed / 2 skipped was preserved exactly,
and again with the new tests, giving 210 / 2.

The two pre-existing skips are unchanged from `ca51de4f`.

Side effect worth noting: because logging is now configured, the failure messages
that tests deliberately provoke are visible in suite output (e.g.
`ERROR: ealpr.inference_jobs: Inference job ... failed [internal_error]: boom`).
Those are **expected** — they are the assertions firing — and were previously
invisible for the opposite reason: nothing was collecting them.

---

## 7. Data integrity

Read-only `_id`-sorted snapshot, taken before implementation, after the test
runs, and again after the subprocess e2e boots:

| Metric | Pre | Post-test | Post-e2e |
|---|---|---|---|
| `detection_result` count | 66 | 66 | 66 |
| `visitors` count | 1023 | 1023 | 1023 |
| `inference_job` count | 0 | 0 | 0 |
| `original_bytes` | 6,355,490 | 6,355,490 | 6,355,490 |
| `processed_bytes` | 946,803 | 946,803 | 946,803 |
| path-backed detections | 6 | 6 | 6 |
| content digest (`_id`-sorted) | `9200c7588833bb5456221036de1d6913` | identical | identical |
| authorized / unauthorized | 812 / 211 | 812 / 211 | 812 / 211 |
| active visitors | 317 | 317 | 317 |

Every value matches the baseline exactly. No writes were performed; the only
database operations were the read-only checks plus the read-only `create_app()`
startup path (index creation, recovery, admin bootstrap check).

C3's orphan-artifact cleanup was observed emitting `Removed abandoned staged
inference artifact` during boots. It was verified to be scoped to `TEMP_DIR`
files matching `^inference_job_[0-9a-f]{32}_.+$` (per `_STAGED_ARTIFACT_RE`) and
touches neither `detected_plates` nor Mongo — consistent with the unchanged
digest and file counts.

---

## 8. Files changed

| File | Change |
|---|---|
| `config.py` | `import logging` + module logger; two model-path `print` → `log.warning`; named constants `DEVELOPMENT_SECRET_FALLBACK`, `KNOWN_DEFAULT_ADMIN_PASSWORD`, `PLACEHOLDER_SECRET_KEYS`; new `assert_production_secrets()` (+66 lines) |
| `ealpr/__init__.py` | `import logging`/`import re`; module logger; `_URI_USERINFO` + `_redact_credentials()`; `_ensure_startup_logging()`; S5 hook in `create_app()`; 10 `print()` → logger calls with mapped levels and redaction |
| `tests/test_secret_logging_docs.py` | **new** — 28 focused tests across S5/S6/S7 |
| `.env.example` | runtime-variable section, `DEFAULT_ADMIN_PASSWORD`, `SECRET_KEY` guidance, S2 note |
| `README.md` | *Runtime Modes and Production Readiness* section (+42 lines) |
| `PHASE2_BLOCK_C5_S5_S6_S7_REPORT.md` | this report |

Not staged: the 9 tracked `__pycache__/*.pyc` files showing modified. Those are
unrelated bytecode churn from running the suite and are never staged.

---

## 9. Out of scope and not done

Untouched, as required:

- YOLO/OCR models, `ealpr/ai_models.py` load diagnostics
- the inference pipeline, `process_plate`/`video`, `_claim()`, `run_job()`,
  `_persist_detection()`, `_complete()`, the queue
- camera code, Socket.IO event handlers, all UI/blueprint code
- repositories, all schemas, image storage, thumbnails
- C2/C3/C4 semantics — `recover_jobs`, worker supervision, monitor loop
- authentication/authorization, password hashing
- Redis, Celery, RQ, Kafka, RabbitMQ
- CI/CD, rate limiting
- **production WSGI server selection (S2)**
- admin creation semantics — `_create_default_admin()` still performs exactly
  the same check-and-create, only its output moved from `print` to `log`. The
  `DEFAULT_ADMIN_PASSWORD == "admin123"` warning branch is unchanged; in
  production that branch is now unreachable anyway, because S5 refuses to start
  while the value is still `admin123`.
- the two model-path warnings remain `WARNING` in `config.py` and are emitted
  only at import, unchanged in frequency.

---

## 10. S2 status — not implemented

**S2 was not implemented, not started, and not partially configured.**

- No production WSGI server was chosen, installed or configured.
- `requirements.txt` was **not** modified; it contains no WSGI server.
- No gunicorn/waitress/uvicorn/hypercorn/gevent/eventlet configuration, script,
  entry point or documentation was added — a test
  (`test_no_wsgi_server_is_documented_as_available`) asserts that neither
  `README.md` nor `.env.example` mentions any of them, so this cannot regress
  unnoticed.
- `README.md` states explicitly that S2 is not yet implemented and that the
  application is not production-ready. A test asserts both claims are present
  (`test_documents_state_that_s2_is_not_implemented`,
  `test_no_production_readiness_is_claimed`).
- The only startup command documented remains `python app.py`.

Choosing, installing and configuring a production WSGI server — and reconciling
that choice with the single-process requirement documented in §4 — remains open
work for a later block.

---

## 11. Risks and limitations

- **S5 validates `SECRET_KEY` and `DEFAULT_ADMIN_PASSWORD` only.** It does not
  validate `MONGODB_URI` credentials, `API_KEY_21ST`, `TWENTYFIRST_TOKEN` or
  model paths. Production remains free to boot with a weak database password.
- **`SECRET_KEY` strength is not measured.** Any non-empty value outside the two
  known placeholders is accepted, including a short or guessable one. There is
  no entropy threshold.
- **The default admin account is created with whatever password is configured.**
  S5 blocks the known default at startup but cannot tell whether an operator
  chose a *different* weak password.
- **The rejection depends on the mode.** Running with `EALPR_ENV` unset (the
  development default) bypasses S5 entirely by design. Nothing detects a
  production deployment that forgot to set `EALPR_ENV=production`.
- **Logging is configured once, by the application.** If a host WSGI server or a
  test harness configures logging first, `_ensure_startup_logging()` defers to
  it (by design). Application log level is fixed at `INFO` and is not
  configurable by environment variable.
- **`_redact_credentials` handles `scheme://userinfo@` shapes only.** A
  credential embedded in some other form (a query parameter, a token in a path)
  would not be caught. The invalid-URI message avoids the problem entirely by
  not logging the input at all.
- **Documentation is prose.** Only the assertions listed in §5 are enforced by
  tests; the rest of the README could drift without failing the suite.
- **S7 did not produce an operations runbook.** Start/stop, log locations, and
  failure triage are described only to the depth needed to document the four
  runtime variables.

---

## 12. Verification commands

```powershell
# focused
.venv\Scripts\python.exe -m unittest tests.test_secret_logging_docs
.venv\Scripts\python.exe -m unittest tests.test_runtime_lifecycle
.venv\Scripts\python.exe -m unittest tests.test_inference_jobs

# full suite
.venv\Scripts\python.exe -m unittest discover -s tests -t .

# data integrity (read-only, _id-sorted)
.venv\Scripts\python.exe <integrity snapshot script>
```

---

## 13. Next steps

Block C5 is complete (S1, S3, S4, S5, S6, S7 committed; S2 deliberately
remaining).

Recommended order for the next block:

1. **S2 — production WSGI server.** The blocking precondition is now documented:
   the server must run a **single** worker with the reloader off, or the
   guarantees in §4 break. Any choice must be reconciled with
   `assert_single_process()`, which currently rejects `EALPR_WORKERS != 1`.
2. Close the S5 limitation on non-`production` deployments (a production host
   that forgets `EALPR_ENV` still boots in development mode with fallback
   secrets).
3. Extend production validation to `MONGODB_URI` credentials and apply it to the
   other secrets (`API_KEY_21ST`, `TWENTYFIRST_TOKEN`).
4. Migrate the remaining runtime `print()` calls (§3) to the logging discipline
   established here.
5. Add a configurable log level and log destination for deployed runs.

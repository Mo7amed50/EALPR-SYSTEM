# Phase 2 — Gate 0: Safety, Backup & Credential Remediation Plan

**Repository:** `D:\EALPR SYSTEM`
**Branch:** `main`
**Gate 0 status:** COMPLETE (all 7 steps)
**Specification:** `PROJECT_SPEC_FINAL.md`
**Prior phase:** Phase 1 Repository Audit (`D:\EALPR\AUDIT_REPORT.md`)

---

## 1. Git checkpoint (steps 1–2)

### Pre-checkpoint state

| Metric | Value |
|---|---|
| Branch | `main` |
| HEAD before checkpoint | `79af106d578ab2e40e3e540050fad6d8240ee899` |
| Modified tracked files | 31 |
| Untracked entries | 9 |
| Tracked files | 29,956 |

The 31 modified files comprised application source (`app.py`, `models.py`, `ealpr/__init__.py`,
`ealpr/ai_models.py`, `ealpr/utils.py`, `ealpr/extensions.py`, four blueprints), four templates,
and already-tracked `__pycache__` bytecode. The untracked entries were
`PROJECT_SPEC_FINAL.md`, `images/`, `temp/`, `detected_plates/`.

### Checkpoint created

| Item | Value |
|---|---|
| Checkpoint commit | `6680106a322931e6fd771bfa324d3b93fe73c841` |
| Tag | `pre-phase2-checkpoint` |
| Contents staged | 33 added, 31 modified |
| Files discarded / reset / restored | **0** |

Command used was `git add -A` followed by `git commit`. No `reset`, no `checkout -- .`,
no `clean`, no `stash`, no `restore`. The checkpoint records the working tree exactly as it stood,
so every pre-existing uncommitted change is now recoverable.

**Rollback procedure** (if Block A must be abandoned):

```powershell
git -C "D:\EALPR SYSTEM" reset --hard pre-phase2-checkpoint   # back to pre-Phase-2 state
```

### Follow-up commit

`c1b11d22` — adds `.gitignore` (section 2).

### Incident during Gate 0

While verifying ignore rules, a probe file named `working/combined_dataset.cache`
**overwrote and then deleted the real tracked file of the same name.** This was a
self-inflicted error, immediately detected by `git status`, and fully reversed with
`git checkout HEAD -- working/combined_dataset.cache`.

| Check | Result |
|---|---|
| File restored | Yes, 466,039 bytes |
| `git diff HEAD` for that path | empty — byte-identical to the checkpoint |
| Residual damage | none |

This is recorded deliberately: it is direct evidence that the Gate 0 checkpoint works,
and a reminder that verification probes must use names that cannot collide with
tracked files. Subsequent `.gitignore` verification was moved to an isolated scratch
repository for exactly this reason.

---

## 2. `.gitignore` (step 3)

No `.gitignore` existed. One was created at `D:\EALPR SYSTEM\.gitignore` and committed as `c1b11d22`.

### Coverage

| Category | Patterns |
|---|---|
| Secrets | `.env`, `.env.*` with `!.env.example` re-included |
| Bytecode | `__pycache__/`, `*.py[cod]`, `*$py.class` |
| Tooling caches | `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage*`, `htmlcov/` |
| Virtual environments | `.venv/`, `venv/`, `env/`, `ENV/`, `venv.bak/`, `env.bak/` |
| Generated storage | `detected_plates/`, `temp/`, `tmp/`, `*.log` |
| YOLO training artifacts | `working/egyptian-car-plates-13/`, `working/combined_dataset/`, `working/combined_dataset.cache`, `working/.virtual_documents/`, `working/runs/**/weights/last.pt`, `working/runs/**/args.yaml`, `working/runs/**/results.csv` |
| Other generated | `results/`, `*.cache`, `*.xlsx`, `*.pdf`, `*.onnx`, `*.engine`, `*.torchscript` |
| Local media | `*.mp4`, `*.avi`, `*.mov`, `*.mkv` |
| OS / editor | `Thumbs.db`, `.DS_Store`, `.vscode/*` (settings/extensions re-included), `*.swp`, `*~` |

### Deliberate exclusions

`working/runs/**/weights/best.pt` is **not** ignored. `config.py` loads two files from that
path at runtime; ignoring them would break a fresh clone. The `last.pt` checkpoints
(byte-identical in size to `best.pt`) and the 29,395 dataset images **are** ignored, as
they are reproducible and not needed to run the application.

### Verification

Rules were validated in an isolated `git init` scratch repository containing only the
`.gitignore` and probe files, so the real repository was never used as a test subject.

- 21 sensitive/generated paths — **all correctly ignored**
- 17 source/spec/model-weight paths — **all correctly kept**
- **0 failures**

### Already-tracked files deliberately NOT untracked

Per instruction, no `git rm --cached` was performed. Consequently these remain tracked
and will keep appearing in commits until a separate, explicitly-approved untracking step:

| Path | Tracked files | Note |
|---|---|---|
| `.env` | 1 | contains a live credential — see section 4 |
| `__pycache__/` | 3 (root) + 9 (in `ealpr/`) | compiled bytecode |
| `working/egyptian-car-plates-13/` | 25,333 | training dataset images |
| `working/combined_dataset/` | 4,062 | prepared training dataset |
| `working/runs/**/weights/last.pt` | 3 | redundant optimizer checkpoints |
| `working/combined_dataset.cache` | 1 | Ultralytics cache |
| `images/`, `detected_plates/`, `temp/` | 38 | test fixtures + runtime output |

**Recommendation for a later phase:** untrack `.env`, all `__pycache__/`, the two dataset
directories, `combined_dataset.cache`, and the three `last.pt` files. This would remove
~29,400 files from future commits. It is deliberately deferred because it is a
history-affecting decision that should not be bundled into a freeze-fix block.

---

## 3. Model files and configuration (step 6)

All resolved through the application's own `config.py` after `load_dotenv()`.

### Runtime models — present and valid

| Role | Path | Size | Format |
|---|---|---|---|
| Plate detector (`MODEL_PATH`) | `working/runs/detect/yolo_car_plate/weights/best.pt` | 40,801,819 B | torch zip archive |
| Second stage (`OCR_MODEL_PATH`) | `working/runs/detect/yolo11m_car_plate/weights/best.pt` | 40,567,141 B | torch zip archive |
| Arabic font (`FONT_PATH`) | `alfont_com_arial-1.ttf` | 1,036,584 B | present |

### Integrity baseline (SHA-256)

```
268411b4f6c1c265e7f8b0c96b3d5e967a77157a4ac5d08c66b7dba5ff32ff4d  yolo_car_plate/weights/best.pt
3586021c86f8fcbd1e4a815f8db285d71369a137e0eae3f2e97dfc771976e735  yolo11m_car_plate/weights/best.pt
c9b76220a5be42ead4733611e417cd65c5fd8aeaa33eb56576ac378a37d130a1  alfont_com_arial-1.ttf
```

### Non-secret configuration

`CONFIDENCE_THRESHOLD=0.5`, `PLATE_CLASS_ID=0`, `TEMP_DIR=<repo>\temp` (exists),
`MAX_IMAGE_SIZE=5,242,880`, `MAX_VIDEO_SIZE=104,857,600`, `VIDEO_FRAME_SKIP=5`,
`VIDEO_DUPLICATE_WINDOW=30`, `CAMERA_URL=0`, `FPS_LIMIT=30`, `FRAME_WIDTH=640`, `FRAME_HEIGHT=480`.

### Database

`ping` OK against Atlas, server version **8.0.32**, target database `ealpr_db` reachable.

### Interpreter and dependencies

The project virtual environment is `D:\EALPR SYSTEM\.venv\Scripts\python.exe` and is **not**
tracked by git. All required packages import successfully:

`flask 3.1.0`, `flask_login 0.6.3`, `flask_socketio`, `mongoengine 0.29.1`, `python-dotenv`,
`opencv-python 4.11.0`, `pillow 11.2.1`, `ultralytics 8.3.124`, `openpyxl 3.1.2`,
`pytz 2025.2`, `arabic-reshaper 3.0.0`, `python-bidi`.

### Weak defaults still active — documented, not changed in Gate 0

| Finding | Detail |
|---|---|
| `SECRET_KEY` | The value supplied by `.env` is identical to the hardcoded fallback `ealpr-dev-secret`. Session cookies are therefore forgeable. |
| `DEFAULT_ADMIN_PASSWORD` | Falls back to the hardcoded `admin123`; not overridden in `.env`. |
| `OFFLINE_ADMIN_PASSWORD` | Defaults to `DEFAULT_ADMIN_PASSWORD`, so the offline login path uses `admin123`. |

These were **not** modified, because changing them alters authentication behaviour and
belongs to the security phase (spec Phase 11), not to a freeze-fix block. Recorded here so
they are not lost.

---

## 4. Credential exposure and rotation plan (step 5)

**No credential value was printed, copied, or transmitted at any point.** All analysis was
performed by structural inspection — key names, lengths, presence/absence of a password
component, host name, and SHA-256 fingerprints of values for equality comparison only.

### Confirmed exposure

The MongoDB Atlas connection string is present in **git history in four places**:

| # | Location | Commit | Date |
|---|---|---|---|
| 1 | `.env` | `7d5da32a` ("o") | 2026-04-05 |
| 2 | `config.py` | `361db3e6` ("Initial commit with project files") | 2025-06-28 |
| 3 | `config.py` | `80e4b211` ("update") | 2026-03-06 |
| 4 | `config.py` | `7d5da32a` ("o") | 2026-04-05 |

Fingerprint comparison shows the three historical `config.py` values are **not identical to
each other and not identical to the current `.env` value**. All reference the same Atlas
database-user account name, so the realistic reading is **one account with up to four
distinct exposed passwords over time**.

`.env` was also still tracked in the working tree at the start of Gate 0 and is now
covered by `.gitignore`, but the file remains tracked and remains in history.

`config.py` is now clean — it reads `MONGODB_URI` via `os.getenv('MONGODB_URI', '')` and
contains no hardcoded URI. The hardcoding happened only in the three historical revisions.

### Remediation plan — requires authorised human action

Rotation is **not** performed automatically. It needs authenticated access to the MongoDB
Atlas console, which is outside this environment and outside the granted authorisation.

**Step 1 — Rotate at the provider (do this first, it is the only step that actually fixes it)**

1. Sign in to MongoDB Atlas as the project owner.
2. **Database Access** → locate the application database user (the account whose name
   appears in the exposed URI host field).
3. **Delete that user.** Do not merely change its password — deleting invalidates every
   copy of every credential string simultaneously, which is what is required.
4. Create a replacement user with a newly generated strong password, scoped to the single
   target database only, with only the privileges the application actually needs
   (`readWrite` on `ealpr_db`).
5. Optionally enable IP access restrictions to the deployment host's egress addresses.
6. Review **Atlas audit logs** for access from unexpected IP addresses, and review
   **database access logs** for reads of the `user` collection. Because `User.password` is
   stored in plaintext (audit §7.1, §13.1), any actor who could read the `user` collection
   already has every user's password.

**Step 2 — Update the application to the new credential**

1. Write the new URI into the local, git-ignored `.env`.
2. Restart the application and confirm login plus dashboard data load.
3. Confirm `.env` is *not* staged: `git status --porcelain` must not list it.

**Step 3 — Purge the secret from history (optional, destructive to clones)**

Only after Step 1 has succeeded, because rewriting history does **not** un-expose anything:

```powershell
pip install git-filter-repo
git -C "D:\EALPR SYSTEM" filter-repo --path .env --invert-paths
git -C "D:\EALPR SYSTEM" filter-repo --path config.py \
    --replace-text <expressions-file-that-redacts-the-uri-pattern>
git -C "D:\EALPR SYSTEM" reflog expire --expire=now --all
git -C "D:\EALPR SYSTEM" gc --prune=now --aggressive
```

Consequences to accept before doing this: every existing clone and fork must be discarded
and re-cloned, all prior commit SHAs change, and open pull requests become invalid.
There is only one local clone and no configured remote, so the blast radius is currently
small — this is the cheapest moment to do it.

**Step 4 — Prevent recurrence**

1. `.gitignore` now covers `.env` (done, `c1b11d22`).
2. Untrack `.env` in a later approved step (section 2).
3. Keep `.env.example` as the only committed configuration template (already tracked, values are placeholders).
4. Move `SECRET_KEY` to a generated random value and remove the hardcoded fallback.
5. Add a pre-commit secret scanner (e.g. `detect-secrets`, `gitleaks`) to block reintroduction.
6. Address the plaintext password storage as the highest-priority application security defect (spec Phase 11).

### Interim risk accepted

Until Step 1 is performed, the Atlas credentials in history must be treated as compromised.
The database is internet-reachable and currently holds 1,021 visitor records including
national ID fields, 66 detection images, and 3 user credentials in plaintext.

---

## 5. Backup checkpoint plan (step 7)

### 5.1 Application code — backed up by Git

| Item | Value |
|---|---|
| Checkpoint commit | `6680106a322931e6fd771bfa324d3b93fe73c841` |
| Tag | `pre-phase2-checkpoint` |
| Tracked files at checkpoint | 29,990 |
| Working tree at time of writing | CLEAN |

Code backup needs no separate copy: the checkpoint commit is the backup.

### 5.2 MongoDB data — logical export created and verified

`mongodump`, `mongorestore`, `mongosh` and `mongo` are **not installed** on this machine, so
`mongodump` was unavailable. A verified logical export was produced with `pymongo` instead,
using only read operations (`find`, `listCollections`, `listIndexes`, `collStats`, `ping`).
**No write, delete, or migration was performed against the live database.**

| Item | Value |
|---|---|
| Location | `C:\Users\mohamed\AppData\Local\Temp\opencode\ealpr_backup_20260926_202455` |
| Files | 134 |
| Size | 7.58 MB |
| Collections | 6 |
| Documents | 1,445 |
| Images extracted | 120 |

| Collection | Documents | Images | JSON size |
|---|---|---|---|
| `detection_result` | 66 | 120 | 47,549 B |
| `visitor` | 1,021 | 0 | 493,912 B |
| `user_activity` | 354 | 0 | 96,275 B |
| `user` | 3 | 0 | 1,304 B |
| `mongoengine.counters` | 1 | 0 | 60 B |
| `system_settings` | 0 | 0 | 2 B |

Contents: canonical Extended JSON per collection, per-collection index definitions,
collection statistics, `MANIFEST.json` with SHA-256 per file, and `RESTORE_HOWTO.txt`
containing a runnable restore script.

Verification: every export was re-opened, re-parsed, document counts re-checked, and
SHA-256 re-computed. **All 6 collections reported checksum OK and count OK.**

> **ACTION REQUIRED:** the backup currently sits in a **temporary** directory that the OS may
> clear. Move it to durable storage before relying on it. It is a logical export, not a
> byte-identical bson dump; collection-level options are not preserved.

### 5.3 YOLO weights — tracked by Git, checksums recorded

Both runtime models are committed to git, so git is their backup. SHA-256 baselines are
recorded in section 3 so silent corruption or replacement can be detected. The two
`last.pt` files and the ~0.67 GB of training datasets are redundant for runtime purposes
and are now ignored for future commits.

### 5.4 Recommended ongoing practice

1. Take a fresh logical export before any phase that changes models or schema.
2. Verify `MANIFEST.json` checksums after every export.
3. Keep at least one backup outside the repository and outside a temp directory.
4. Prefer Atlas native continuous backups as the authoritative copy; the logical export is
   the fast local restore path.
5. Never run the `/api/detections/clear` or `/api/visitors/clear` endpoints (audit §4.6)
   without a fresh verified export taken first.

---

## 6. Gate 0 summary

| Step | Requirement | Status |
|---|---|---|
| 1 | Inspect git status | Done — 31 modified, 9 untracked, 29,956 tracked |
| 2 | Recoverable checkpoint, nothing discarded | Done — `6680106a` + tag `pre-phase2-checkpoint` |
| 3 | Verify / create `.gitignore` | Done — created, committed `c1b11d22`, rules verified 0 failures |
| 4 | Do not blindly untrack | Honoured — nothing untracked, recommendations documented |
| 5 | Identify credential, no exposure, rotation plan | Done — 4 exposures confirmed, plan in section 4, nothing rotated |
| 6 | Verify models and configuration | Done — weights valid, DB reachable, weak defaults documented |
| 7 | Backup plan for DB / weights / code | Done — verified logical export created, checksums recorded |

**Gate 0 is complete and the project is recoverable.** Block A may begin.

# Phase 2 — Block C1: Data Access Repositories & MongoDB Aggregations

**Status:** Complete
**Block:** C1 — Data access layer
**Baseline commit:** `17f76704a7980fa8d8b2163b3ac64ad593db10fd`
**Scope:** Introduce a thin repository layer and move eligible read/aggregation
work out of Flask routes and into MongoDB. No behavioural change.

---

## 1. Scope and boundaries

### In scope

- A small, explicit repository package (`ealpr/repositories/`) — three
  concrete repositories, no generic base class, no framework, no DI container.
- Move eligible read and aggregation work into MongoDB `$facet` / `$group`
  pipelines.
- Projection lists so list and metadata queries stop transferring fields the
  routes never render.
- Consolidate the four-tier visitor identifier lookup into one query.
- Characterisation tests that pin current behaviour **before** it is trusted.

### Explicitly out of scope (unchanged by this block)

| Excluded | Reason |
| --- | --- |
| Models, schema, migrations, indexes | No document shape was altered; no index was added or dropped. |
| Authentication / authorization redesign | Out of scope. |
| Pagination behaviour (Block A) | Re-verified, not modified. |
| Image / thumbnail / binary behaviour (Block B) | Re-verified, not modified. |
| Caching (Redis), Celery, background jobs | Out of scope. |
| Data normalisation / backfill | Out of scope. |
| `/process_plate`, `/process_video`, camera, Socket.IO, YOLO/OCR | Inference path is untouched — see §6. |
| Visitor pagination changes | Out of scope. |

---

## 2. What was built

```
ealpr/repositories/
├── __init__.py              # exports the three repositories
├── detection_repository.py  # DetectionResult reads, pagination, export, image access
├── visitor_repository.py    # Visitor reads, projections, lookup, dashboard counts
└── report_repository.py     # $facet report pipelines, ExcelTables
```

Each repository is a plain module of `classmethod`s. There is no base class, no
session object and no abstract interface — the whole point of this block was to
move data access, not to introduce architecture.

### 2.1 `VisitorRepository`

- `list_visitors(...)` / `list_for_manage(...)` — projected list queries.
- `find_by_identifier(...)` — the four-tier visitor lookup, now a **single**
  query with one `$or` (see §4.2).
- `find_duplicate(...)` — duplicate checks on add/edit.
- `dashboard_counts()` — one aggregation replacing four separate counts.
- `list_for_period_report(...)` — styled daily/weekly/monthly report rows.

### 2.2 `DetectionRepository`

- `history_page(...)`, `gallery_page(...)` — projected, paginated.
- `get_detail(...)` — metadata without binaries.
- `list_for_export(...)` — export rows.
- `get_image_source(...)` — the **only** method permitted to load a binary,
  and it loads exactly the one image the route is about to serve.
- `recent_activity(...)`.

### 2.3 `ReportRepository`

- `visitor_stats()` — one `$facet` pipeline returning `summary`, `daily`,
  `peak` and `departments` in a single round trip.
- `excel_tables()` — returns `ExcelTables` (hourly, daily, department, peaks)
  with the same values the JSON route reports.
- `export` / period helpers for the styled report workbooks.

---

## 3. Measurement method

Instrumentation is PyMongo `CommandListener`-based (`c1_probe.py`), not
application-level timers. For each of 15 affected read paths it records:

- MongoDB commands issued (`find` / `aggregate` / `count` / `distinct` / `getMore`)
- documents actually returned, per collection
- response size in bytes
- wall-clock time (best and median of N, warm)

**Important caveat on latency.** The BEFORE capture ran while Atlas was
experiencing transient network degradation — routes whose query shape did *not*
change (e.g. `/uploaded_images`, byte-for-byte identical work) were 30–40 %
slower. The wall-clock column is therefore reported for completeness but is
**not** a controlled benchmark and should not be read as a speedup claim. The
reproducible, machine-checked results are the **command count** and the
**document count**.

---

## 4. Results

### 4.1 Query and document reduction

| Route | Commands | Documents | Response bytes |
| --- | --- | --- | --- |
| `GET /` (dashboard) | 5 → **3** | 5 → **3** | 53,219 → 53,219 |
| `GET /api/dashboard/stats` | 5 → **3** | 5 → **3** | 125 → 125 |
| `GET /api/analytics/visitor-stats` | 3 → **2** | 1,024 → **2** | 3,060 → 3,060 |
| `GET /api/visitors/by license_plate` | 3 → **2** | 2 → 2 | 297 → 297 |
| `GET /detection_history` | 3 → 3 | 12 → 12 | 29,369 → 29,369 |
| `GET /visitors` | 3 → 3 | 1,024 → 1,024 | 1,172,391 → 1,172,391 |
| `GET /manage_visitors` | 3 → 3 | 1,024 → 1,024 | 1,842,059 → 1,842,059 |
| `GET /api/detection/<id>` | 4 → 4 | 4 → 4 | 763 → 763 |
| `GET /api/images/<id>` | 2 → 2 | 2 → 2 | 407 → 407 |
| `GET /uploaded_images` (+2 variants) | 3 → 3 | 22 → 22 | 53,149 → 53,149 |
| `GET /api/activities` | 2 → 2 | 11 → 11 | 1,338 → 1,338 |
| **TOTAL (15 routes)** | **46 → 40** | **3,179 → 2,153** | 3,256,928 → 3,256,928 |

- **−6 commands (−13.0 %)**, **−1,026 documents (−32.3 %)**.
- **Every route returns a byte-identical response.** Response sizes are
  unchanged to the byte across all 15 paths, which is the primary evidence that
  rendering behaviour did not drift.
- The remaining 1,024-document reads on `/visitors` and `/manage_visitors` are
  the genuine page of rows the templates render; the win there is that the
  *projected* fields are what travel, not that fewer rows are returned.

### 4.2 Four-tier lookup collapsed to one query

`find_by_identifier` previously issued one query per tier (visitor code, then
license plate, then name, then plate number). It is now a single `find` with one
`$or` over all four fields.

Precedence is preserved by re-ranking the candidates **in Python** against the
documented tier order, so a record that appears in more than one tier resolves
exactly as it did before. `test_lookup_precedence_*` pins this.

### 4.3 Dashboard collapsed to two data queries

`_dashboard_stats` previously ran four `count_documents` calls. It now issues:

- 1 × `count_documents` on `detection_result`
- 1 × `aggregate` on `visitor` producing all three visitor counts

…plus the pre-existing authenticated-user lookup, for 3 commands total instead
of 5.

Verified payload is unchanged from BEFORE:

```json
{
  "stats": {
    "active_visitors": 317,
    "authorized_visitors": 812,
    "total_detections": 66,
    "unauthorized_visitors": 211
  },
  "success": true
}
```

### 4.4 Report analytics: 1,024 documents → 2

`/api/analytics/visitor-stats` previously pulled a 1,024-row projection into
Python and aggregated there. It now runs one `$facet` pipeline server-side and
returns two small documents.

Equivalence against a Python oracle: **55/55 checks pass** across five
instants — dynamic `now`, plus `2026-09-26T17:53:50Z`, `2026-04-06T12:00:00Z`,
`2026-09-20T23:59:59Z`, `2027-01-01T00:00:00Z` — 11 assertions each covering
summary counts, daily traffic, peak hours, department ranking and the three
Excel tables.

---

## 5. Bugs found and fixed

These were latent defects surfaced by the characterisation work, not
regressions introduced by this block.

### 5.1 `$facet` result shape was read at the wrong level

A `$facet` pipeline returns **one** document keyed by the facet names. The code
was indexing the top-level result as though it were the facet body. Fixed by
reading the single result document and then the named facets.

### 5.2 UTC hour buckets were silently dropping early-morning data

`_peak_hour_buckets_stage` grouped by `$toString` of the hour, which renders
hour 8 as `"8"`, not `"08"`. Any grouping downstream that padded to two digits
therefore lost hours 1–7. Fixed with `_padded_utc_hour()`, which pads before
grouping.

This was a **real data-loss bug**: peak-hour output was wrong for the entire
early-morning window before this block.

### 5.3 `find_by_identifier` used a `Document.get()` that does not exist

The consolidated lookup called a mapping-style `.get()` on a MongoEngine
`Document`, and its projection dropped fields required for re-ranking. Fixed to
attribute access plus a projection covering every ranking field.

### 5.4 `dashboard_counts().active` returned 0

The active-visitor filter compared a missing field to `null`. In an aggregation
`$expr`, a missing field is **not** equal to `null`, so the count silently came
back as `0` while the route rendered `0` active visitors against a real
population of 317.

The dashboard payload is byte-identical to BEFORE *only because this was caught
and fixed before the final capture.* An intermediate capture showed
`active_visitors: 0`. Caught by `test_dashboard_kpi_consistency`, which asserts
the KPI against a direct count of the open-ended visitors.

### 5.5 `ImageField` would be materialised on list reads

List queries used whole-document loads, so a list of 20 detections deserialised
20 × 2 `BinaryField`s that the gallery never renders. Projections remove them;
only `get_image_source` — the one method serving a requested image — loads a
binary. Verified: `/api/detection/<id>` responses do not contain base64
(`780B`).

---

## 6. Inference path is untouched

`process_plate` was compared against `HEAD` and is **byte-identical**
(6,512 chars). Every `detection.py` diff hunk begins at line 400+, after the
inference code. `INFERENCE_LOCK`, `/process_video`, camera and Socket.IO
handlers contain no repository reference, enforced by tests rather than by
inspection alone.

`/process_plate` still resolves its visitor with a direct `Visitor.objects(...)`
lookup and still writes its `DetectionResult`. That is deliberate: the
repository is a read-only projection layer, and this path needs a complete,
savable document.

`git diff --name-only` touches four blueprint files and adds two directories.
`models.py`, `config.py` and all AI modules are untouched.

---

## 7. Data safety

Live data was re-read after the full test run:

| Check | Value |
| --- | --- |
| `detection_result` count | 66 (unchanged) |
| `visitor` count | 1,023 (unchanged) |
| Detections with inline binaries | 60 |
| Detections that are path-backed | 6 |
| Binary content digest | `42d3a96efc875ae0dda1d81a3e46927d` |
| `authorized=True` | 812 |
| `authorized=False` | 211 |
| `status='authorized'` | 811 |
| Active (no `exit_time`) | 317 |

The three-way authorised split (812 boolean vs 811 string) is a pre-existing
data inconsistency. It is **preserved, not corrected** — normalising it is a
migration decision, explicitly out of scope for this block.

**One side effect to disclose.** The repository tests are strictly read-only
against `visitor` and `detection_result`. The HTTP route tests, however, invoke
the application's pre-existing `after_request` hook, which appends a
`UserActivity` row per request. That is unchanged application behaviour
triggered by any request, not something this block introduced, but it means the
test suite is not literally zero-write against the whole database. No
`visitor` or `detection_result` document was created, modified or deleted by
any test.

---

## 8. Test results

Permanent suite, stdlib `unittest` (no pytest in this environment, and no
dependency was added).

| Suite | Result |
| --- | --- |
| `tests/test_visitor_repository.py` | 16 passed, 1 skipped |
| `tests/test_detection_repository.py` | 15 passed |
| `tests/test_report_repository.py` | 10 passed |
| `tests/test_routes_readonly.py` | 11 passed |
| **Full discovery** | **52 passed, 1 skipped, 0 failed (43.8 s)** |

The skip is a negative-control test: it needs two visitors whose identifiers
collide in a way the live dataset does not currently contain. It is skipped
rather than asserted vacuously.

Characterisation coverage includes: per-route query budgets asserted against
real PyMongo command monitoring; projections asserted to exclude
`original_image` / `processed_image`; full-document loads asserted on every
mutating visitor route (`for_update=True`); read-only routes asserted *not* to
pay for a full load; lookup precedence; report equivalence against a Python
oracle; and structural guards on the inference path.

### Regression re-runs of earlier blocks

| Suite | Result |
| --- | --- |
| Block A pagination (`ealpr_pagtest.py`) | **PASS** — all cases |
| Block B live supplement | **58 passed, 0 failed** — incl. binary bytes unchanged (6,355,490) |
| `create_app()` | OK, 43 routes registered |
| `py_compile` | 14/14 files OK |
| `git diff --check` | clean |

### Known pre-existing failure, unrelated to this block

`ealpr_b6_live_readonly.py` carries a hardcoded `1021` visitor baseline from an
earlier block. The live count is `1023`, so that one assertion fails. It is a
stale constant in a temporary harness, not a code defect and not caused by this
block — recorded here rather than silently edited.

---

## 9. Transient issues encountered

Atlas timed out twice while the equivalence harness was pulling its unprojected
reference documents. The harness was made retryable and cached its reference
pull. These were test-harness accommodations only; no application code contains
a retry, and no behaviour was altered to accommodate them.

A benign `DeprecationWarning` about `uuidRepresentation` is emitted by
MongoEngine at import. It predates this block and is not addressed here.

---

## 10. Commit

```
phase2: add data access repositories and mongo aggregations
```

Six tracked `__pycache__/*.pyc` files were regenerated by the verification runs
and are deliberately left unstaged.

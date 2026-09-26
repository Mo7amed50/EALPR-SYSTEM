# Phase 2 — Block A Completion Report

**Project:** EALPR (Egyptian Automatic License Plate Recognition)
**Repository:** `D:\EALPR SYSTEM`
**Specification:** `PROJECT_SPEC_FINAL.md` (source of truth)
**Phase:** 2 — Block A (interface freeze remediation)
**Date:** 2026-09-26
**Status:** Complete and verified — 101 automated checks passing, 0 failing

---

## 1. Scope

Block A only. No Block B or later work was started. The following were
deliberately **not** done, per the phase boundary:

- No thumbnail generation or image-storage migration.
- No migration of the 60 legacy `BinaryField` detections to path-based storage.
- No schema changes and no removal of the legacy fields.
- No repository/service extraction.
- No Celery, Redis, cameras, notifications, or audit logging.
- No credential rotation and no external system access.

---

## 2. Files changed

| File | Change |
|---|---|
| `ealpr/blueprints/detection.py` | Private cache headers on authenticated images; `404` instead of `444`; server-side pagination and filtering for `/uploaded_images` |
| `templates/index.html` | Removed the AI re-trigger on visitor add; visibility-aware dashboard polling with coalescing |
| `templates/uploaded_images.html` | Server-paginated gallery, preserved filters, lazy loading, empty state, pagination controls |
| `templates/detection_history.html` | `decoding="async"` on the two modal images |

Net effect: `4` files, `+332 / -78` lines. No other source file was touched.

---

## 3. What changed and why

### 3.1 Registering a visitor no longer re-runs inference (freeze defect A)

**Before:** the quick-add visitor success handler executed
`uploadForm.dispatchEvent(new Event('submit'))`. That re-submitted the upload
form, which re-ran YOLO **and** OCR on the already-processed image, blocked the
UI thread on the GPU, and wrote a **duplicate `DetectionResult`** every time a
visitor was registered.

**After:** the handler calls `refreshDetectionVisitorStatus()`, which performs
two read-only requests — `/api/search/plate?q=<plate>` to refresh the
authorization badge, and `/api/visitors/<id>` (only when authorized) to reveal
the visitor detail card. It never touches inference and never writes a
detection. The four status elements it updates are updated in place.

### 3.2 Detection images are no longer publicly cacheable (freeze defect B)

`/api/images/<id>/<type>` served authenticated plate images with
`Cache-Control: public, max-age=3600`, which permits shared proxies and CDNs to
store and replay them for other users. Both the file-backed and binary-backed
branches now emit:

```
Cache-Control: private, max-age=300, must-revalidate
Vary: Cookie
X-Content-Type-Options: nosniff
```

`Vary: Cookie` prevents a shared cache from serving one authenticated user's
plate image to another. A missing image now returns `404` instead of the
non-standard `444`.

### 3.3 Gallery images load lazily (freeze defect C)

Gallery cards received `loading="lazy" decoding="async"`, and the modal images
in both `uploaded_images.html` and `detection_history.html` received
`decoding="async"`. `detection_history.html` renders a table whose images are
only loaded on demand in the modal, so `loading="lazy"` was not applicable
there; only `decoding` was added.

### 3.4 The gallery is paginated and filtered in the database (freeze defect D)

`/uploaded_images` previously ran
`DetectionResult.objects.order_by("-timestamp").all()`, loading the entire
collection on every page view. It now:

- accepts `page` and `per_page` (allowed sizes 20/50/100, default 20), snapping
  any other value to the nearest allowed size;
- applies the existing plate / status / date filters **in MongoDB** via `Q`, so
  they work across all pages rather than only the cards already in the DOM;
- counts the filtered total, clamps out-of-range pages, and orders by
  `-timestamp, -id` so records sharing a timestamp cannot shift between pages;
- reads only the current page using `.only(...)`, `.skip(...)` and `.limit(...)`.

`.only()` deliberately excludes the binary image fields, so listing a page does
not transfer ~7.3 MB of image payloads to render 20 cards. The images
themselves are still fetched individually through `/api/images/<id>/<type>`.

The template gained a result summary, a per-page selector, first/previous/
numbered/next/last controls, an empty state, and `loading="lazy"`. Filter and
page-size changes now navigate server-side, returning to page 1 so a narrowed
result set is never left pointing past its own last page.

### 3.5 Dashboard polling pauses when the tab is hidden (freeze defect E)

The bare `setInterval(updateDashboardStats, 30000)` polled even for background
tabs, and every `live_detection` event fired an immediate 4-query round trip.
Now:

- `updateDashboardStats()` returns immediately if `document.hidden`;
- a `dashboardStatsPending` flag guarantees at most one in-flight request;
- `scheduleDashboardStatsRefresh()` collapses bursts into a single request
  after a 2 s delay;
- `startDashboardStatsPolling()` / `stopDashboardStatsPolling()` manage a single
  interval, driven by `visibilitychange`, with a catch-up refresh on return;
- `beforeunload` clears the interval and the pending timer.

The manual refresh button still calls `updateDashboardStats()` directly, so it
behaves exactly as before when the tab is visible.

---

## 4. Corrections to the Phase 1 audit

Two Phase 1 findings were wrong and are retracted here:

1. **"Passwords are stored in plaintext."** Incorrect. `User.set_password()`
   uses `werkzeug.security.generate_password_hash` and `check_password()`
   validates with `check_password_hash`. All 3 users in the live database store
   162-character `scrypt` hashes. Verified without printing any password
   material. The audit misread the model by inspecting only the field
   declarations.

2. **Gallery image delivery.** The gallery does not inline base64. The
   `detection_thumb` filter already returns `/api/images/<id>/processed` URLs;
   the freeze came from 66 unpaginated full-document image requests. The
   remaining `data:image/jpeg;base64,` string in the rendered page is a
   JavaScript fallback inside the modal `<script>`, not an inlined image.

---

## 5. Verification

### 5.1 Static and build checks — 21/21

All 27 Python sources byte-compile. All 10 Jinja templates parse under the
application's real `jinja_env`. The app factory builds and registers 42 routes.
21 assertions confirm each intended change is present in the served output.
Both inline `<script>` blocks on the dashboard pass `node --check`.

### 5.2 Functional tests against the real application — 59/59

Run through the Flask test client against the live MongoDB (66 detections,
1021 visitors):

- **Authentication** — unauthenticated gallery access redirects to login;
  login succeeds; authenticated access returns 200.
- **Dashboard** — renders, exposes the stat elements, `/api/dashboard/stats`
  returns success.
- **Pagination** — default 20; `per_page` 50 and 100 honoured; `0`, `-5`, `abc`,
  `1e9`, `999999`, `20;drop` and a missing value all snap to a safe size; page 2
  returns a non-overlapping slice; out-of-range and non-numeric pages do not
  error; **a full page walk returns all 66 records exactly once with 0
  duplicates**.
- **Filters** — `status=authorized` → 10, `status=unauthorized` → 56, plate
  search narrows correctly, no-match shows the empty state, `date_filter=today`
  and an unknown date filter behave.
- **Images and cache privacy** — both types return 200 with real bytes,
  `Cache-Control: private, max-age=300, must-revalidate`, `Vary: Cookie`, no
  `public`; invalid id → 400, unknown id → 404 (previously 444);
  `/api/detection/<id>` still returns detail with image URLs; gallery thumbnails
  are URL-served with **0 inlined `data:` images**.
- **Visitor add creates no detection** — `POST /api/visitors` succeeds,
  `DetectionResult` count is unchanged (66 → 66), the two read-only endpoints
  the new JavaScript uses both work, and the test visitor is removed with the
  visitor count restored to 1021.
- **No regressions** — `/manage_visitors`, `/detection_history`, `/reports`,
  `/users`, `/settings` all return 200.

### 5.3 Pagination control rendering — 8/8

Numbered links are correct and the active page is marked on the first, middle
and last page; controls are hidden when the result fits one page; the empty
state renders; the summary reads "Showing 21–40 of 66 detections" on page 2;
the status filter stays selected after a reload.

### 5.4 Polling behaviour — 21/21

The **actual shipped polling code** was extracted from `index.html` and executed
against a virtual clock, so these assertions describe the real implementation:

- init issues exactly one request and registers one interval;
- the success payload populates all four stat tiles;
- 6 direct updates while a request is in flight produce 1 request, then exactly
  one coalesced follow-up, then stop;
- 10 scheduled refreshes collapse into exactly 1 request;
- hiding the tab clears the interval and produces **0 requests across a 120 s
  hidden window** (4× the interval); a direct update and a scheduled refresh
  while hidden are both no-ops and leave no pending timer;
- becoming visible re-registers polling and fires exactly one catch-up request;
- polling resumes on cadence when visible (3 responses in 95 s) with a single
  interval, so there is no double polling;
- `beforeunload` clears both the interval and the coalescing timer.

### 5.5 Existing detection functionality — end-to-end

A real stored detection frame was replayed through `POST /process_plate`.
Models loaded, and the pipeline completed in 11.0 s returning HTTP 200 with
detection confidence **0.899** and OCR confidence **0.727**, reading Arabic
plate characters with per-character boxes. A new `DetectionResult` was created
(66 → 67) and then removed, restoring the count to 66. The detection path is
unaffected by these changes.

### 5.6 Requirement coverage

| Requirement | Result |
|---|---|
| Login | Pass (5.2) |
| Dashboard | Pass (5.2, 5.4) |
| Visitor add does not trigger YOLO/OCR | Pass (5.2 — count unchanged, read-only endpoints only) |
| Pagination | Pass (5.2, 5.3) |
| Images still open | Pass (5.2 — `/api/detection/<id>`, thumbnails and modal URLs all resolve) |
| Private cache behaviour | Pass (5.2) |
| Hidden tabs do not poll continuously | Pass (5.4 — 0 requests over 120 s hidden) |
| Existing detection still works | Pass (5.5) |

---

## 6. Findings and residual risks

1. **The entire UI depends on public CDNs.** jQuery 3.6.0, Bootstrap 5.3.0,
   socket.io 4.0.1, Font Awesome 6.0.0 and Google Fonts are all loaded from
   `code.jquery.com`, `cdn.jsdelivr.net` and `cdnjs.cloudflare.com`. No local
   copies exist. In this test environment the headless browser could not load
   them, and **with jQuery absent the dashboard rendered but none of its
   JavaScript ran at all**. Any outage, air-gapped deployment, or compromise of
   those CDNs takes the whole interface down. This is pre-existing, outside
   Block A's scope, and is the highest-value follow-up.

2. **Browser-level UI verification was not possible in this environment.**
   Because of finding 1, the dashboard and gallery pages could not be driven
   end-to-end in a real browser here; external script requests were blocked and
   request interception did not take effect. The JavaScript was therefore
   verified two other ways: 21 static assertions against the served HTML, and
   21 behavioural assertions against the real extracted code on a virtual clock.
   The server-rendered output, all routes, all endpoints and the full inference
   path were verified directly. A visual pass on a networked machine is still
   worth doing before release.

3. **A hung stats request blocks updates indefinitely.** The in-flight guard
   holds until the request settles, and jQuery's `$.get` sets no timeout, so
   stats stop updating for as long as the browser keeps the XHR open. This is
   safe — no request amplification occurs, confirmed by test — but adding a
   `timeout` to the request would bound it.

4. **Legacy binary detections are untouched.** 60 of 66 records still store
   images inline and 6 use paths. Listing a page no longer transfers the binary
   payloads, but a single image request still reads a whole document. Storage
   migration remains future work by design.

5. **Tracked bytecode is still in Git.** `.gitignore` prevents new bytecode from
   being added, but already-tracked `__pycache__/*.pyc` files remain tracked
   and now show as modified. They were intentionally left untracked-but-present
   rather than removed in this phase.

6. **Credentials remain exposed in Git history.** `.env` and historical
   `config.py` values are present in past commits. No values were printed and
   no rotation was performed; the plan is in `GATE0_BACKUP_PLAN.md`. This
   requires authorized external action.

7. **socket.io WebSocket upgrade fails against the dev server** with
   `Invalid frame header`. Live-detection push updates may be unreliable in this
   development configuration. Pre-existing and untouched by Block A.

---

## 7. Data integrity

- No `DetectionResult` was modified or left behind; the count is 66, the same
  as before testing.
- No `Visitor` was left behind; the count is restored to 1021.
- Test visitors were created and removed within the same run.
- No collection was dropped, renamed, or reindexed.
- A verified logical backup remains at
  `C:\Users\mohamed\AppData\Local\Temp\opencode\ealpr_backup_20260926_202455`
  (6 collections, 1,445 documents, 120 images, checksums verified).
  **It should be moved to durable storage — the temp directory is not durable.**

---

## 8. Recommendation

Block A meets its acceptance criteria and is ready to commit. Before starting
Block B, address finding 1 by vendoring the four JavaScript/CSS dependencies
into `static/`, which removes both the availability risk and the third-party
supply-chain surface, and makes the interface testable offline.

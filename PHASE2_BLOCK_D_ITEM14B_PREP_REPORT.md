# Phase 2 / Block D, Item 14B - Secure Backup & Migration Preparation

Scope: **preparation only**. No migration was executed. No document was inserted,
updated, or deleted in MongoDB. No credential was rotated. Nothing was staged,
reset, cleaned, or committed.

Baseline integrity (read-only, before and after this task):
`content_digest=9200c7588833bb5456221036de1d6913` (unchanged),
`det_count=66`, `visitor_count=1023`, `inference_job_count=0`.

## Part A - Durable, access-controlled copy of the verified backup

* Verified logical export: `ealpr_backup_20261014_210334` (7 collections,
  1528 documents, 23 files: 7x `.json` + 7x `.indexes.json` + 7x
  `.collstats.json` + `MANIFEST.json` + `RESTORE_HOWTO.txt`).
* Copied to a durable location outside the repo and the temp workspace:
  `C:\Users\mohamed\EALPR_backups\ealpr_backup_20261014_210334`.
* Durable copy independently re-verified against `MANIFEST.json`
  (SHA-256 + size + document count per file): **`VERIFY_RESULT=ALL_OK`**.
* Filesystem permissions (`Get-Acl`):
  * Owner: `DESKTOP-1FI2KTL\mohamed`
  * `NT AUTHORITY\SYSTEM` - FullControl (inherited)
  * `BUILTIN\Administrators` - FullControl (inherited)
  * `DESKTOP-1FI2KTL\mohamed` - FullControl (inherited)
  * No `Users` / `Everyone` / `Authenticated Users` entry - no broad read access.
* The original temp copy was left in place and is still present:
  `...\Temp\opencode\ealpr_backup_20261014_210334`.

Result: the backup is verified in two locations, and the durable copy is readable
only by the owning user, administrators, and the system account.

## Part B - Cause of the user-count discrepancy (35 now vs 3 at GATE 0)

Read-only investigation. No usernames, emails, hashes, or personal data were
printed.

* Server holds only three databases: `admin`, `ealpr_db`, `local`. The only
  `user` collection is `ealpr_db.user` (count 35). There is no second database or
  collection with a count of 3 - so the snapshot did **not** come from a
  different store.
* `created_at` distribution:
  * `2026-04-05` -> **3** users  (admin + 2 pre-existing accounts)
  * `2026-10-14` -> **32** users
* Username-prefix census (counts only):
  * `__c2_owner_*` = 14
  * `__c2_other_*` = 18
  * `__c2_*` total = **32**
  * `admin` = 1, other pre-existing = 2

**Conclusion:** the 32 extra users are orphaned test fixtures, not real
accounts. `tests/test_inference_jobs.py` (`setUp`, lines ~124-134) creates two
users named `__c2_owner_<hex>` / `__c2_other_<hex>` with the literal password
`"pw"` (assigned directly to the field, so it is **never** passed through
Werkzeug and stays plaintext), and removes them in `addCleanup(self._cleanup)`.
When a test run is killed hard (e.g. the shell timeout during the earlier suite
runs), `addCleanup` never runs and the two users leak. 16 interrupted runs x 2 =
32 - matching the census and the observed plaintext length of 2. This is the same
kill-before-cleanup incident that previously produced orphaned
`__c2_test_*` detection/job documents. The mismatch is therefore an artifact of
test leakage, not of the GATE 0 snapshot being wrong and not of 32 genuinely
provisioned accounts.

Note: these leaked rows are what inflate the "possible plaintext" set to 32. A
real user population is 3 (1 admin + 2 accounts, all already Werkzeug hashes).

## Part C - Dry-run-only migration prototype + synthetic tests

New file `scripts/migrate_passwords.py` (prototype; not wired into the app, never
run against the live DB):

* Pure helpers `is_recognized_hash`, `plan`, `migrate` operate on record dicts.
* Classification is **format-based and conservative**: a value is untouched only
  if it matches Werkzeug's exact output shape (`pbkdf2:`/`scrypt:` + exactly two
  `$`). Short values are not assumed plaintext; long values are not assumed
  hashes.
* `migrate(..., apply=False)` is the default and returns copies unchanged
  (rehearsal). `apply=True` replaces only non-recognised, non-empty passwords
  with `generate_password_hash` and copies every other field verbatim.
  Idempotent after a real apply.
* CLI defaults to dry-run. `--apply` writes **only** when
  `EALPR_MIGRATION_AUTHORIZED` equals the agreed token; otherwise it exits with
  status 2 and a refusal message. Until that authorization mechanism is formally
  defined, the write path is unreachable.
* No password value, salt, or hash is ever printed or logged.

New file `tests/test_password_migration.py` (synthetic data only, no database):
12 tests, all passing (`Ran 12 tests ... OK`). Coverage:

* plaintext -> hash, and the hash verifies for the right password / rejects a
  wrong one;
* a real Werkzeug hash is never hashed a second time;
* dry-run changes nothing;
* migration is idempotent;
* `_id`/username/`is_admin`/`is_active` and record count are preserved;
* classification is format-driven, not length-driven; a colon-bearing non-hash
  is not mistaken for a hash;
* no synthetic password or hash appears in stdout, stderr, or logging output;
* `--apply` without the authorization token is refused.

## Part D - Restore-test prerequisites

`RESTORE_HOWTO.txt` (in the backup) already prescribes the correct procedure:
verify first, drive the restore with pymongo + `bson.json_util`, and restore into
an **isolated, non-production** database - never the live `ealpr_db`.

A safe restore test can be performed without touching `ealpr_db`, because a
distinct database name on the same server (e.g. `ealpr_db_restore_test`) is
logically isolated from `ealpr_db`; a separate MongoDB instance is an even
stronger isolation. What is **missing** for a repeatable, evidence-producing
test:

1. An isolated target that is created and dropped by the harness itself - not
   created by hand - so the live `ealpr_db` is provably untouched.
2. A restore driver script (the HOWTO is documentation only; no
   `d14a_restore.py` exists). It should also recreate the indexes from the
   `.indexes.json` files.
3. A post-restore comparator that diffs the restored collection counts (and,
   ideally, the content digest) against `MANIFEST.json` / the live baseline, so
   "the backup restores correctly" is a checked fact rather than an assumption.
4. Confirmation that the configured MongoDB credentials are permitted to create
   and drop a scratch database (credential scope is currently unknown).

Per this task's constraints, no restore target was created and no restore was
performed.

## Files

* Added (untracked, not staged): `scripts/migrate_passwords.py`,
  `tests/test_password_migration.py`, this report.
* Unchanged and preserved: the nine pre-existing modified `.pyc` files.
* Backup: durable copy at `C:\Users\mohamed\EALPR_backups\ealpr_backup_20261014_210334`
  (verified `ALL_OK`); temp copy retained.

## Recommended next actions (need authorization)

1. Decide how to treat the 32 leaked `__c2_*` test users (delete vs. leave) and
   harden `test_inference_jobs.py` cleanup so a kill cannot leak users.
2. Have the user formally define the migration authorization mechanism, then run
   the prototype dry-run and review the plan counts before any apply.
3. Implement the isolated restore driver + comparator (Part D items 2-3) and run
   it against a scratch database.

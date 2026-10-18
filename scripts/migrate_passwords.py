"""Password-hash migration prototype (Phase 2 / Block D, Item 14B).

PROTOTYPE ONLY. This module is not wired into the application and has never been
run against the live database.

Design rules
------------
* Pure, database-agnostic helpers (``is_recognized_hash``, ``plan``, ``migrate``)
  operate on plain record dictionaries, so they can be exercised with synthetic
  data and no connection.
* The CLI **defaults to dry-run**. It refuses to write unless an explicit future
  authorization mechanism is supplied (``EALPR_MIGRATION_AUTHORIZED`` set to the
  token below). Until that mechanism is formally defined, the write path is
  deliberately unreachable.
* Classification is **conservative and format-based**, never length-based. A
  value is left untouched only if it matches the exact shape Werkzeug's
  ``generate_password_hash`` emits (``pbkdf2:``/``scrypt:`` + two ``$``). A short
  value is not assumed to be plaintext, and a long value is not assumed to be a
  hash.
* No password value, salt, or hash is ever printed or logged.
"""
import argparse
import os
import re
import sys

from werkzeug.security import generate_password_hash

# Werkzeug emits "<method>$<salt>$<hash>" with method "pbkdf2:<algo>:<n>" or
# "scrypt:<n>:<r>:<p>". Requiring the known prefixes AND exactly two '$' avoids
# treating an arbitrary string that merely contains ':' as a hash.
_HASH_RE = re.compile(r"^(?:pbkdf2|scrypt):[^$]*\$[^$]*\$[^$]+$")

AUTHORIZATION_ENV = "EALPR_MIGRATION_AUTHORIZED"
AUTHORIZATION_TOKEN = "I_ACCEPT_PASSWORD_MIGRATION"

# Fields that must survive a migration untouched.
_PRESERVED_FIELDS = ("_id", "username", "is_admin", "is_active")


def is_recognized_hash(value):
    """True only for a value in Werkzeug's own hash format."""
    return isinstance(value, str) and bool(_HASH_RE.match(value))


def plan(records):
    """Summarise a migration without mutating or echoing anything.

    Returns counts only; no value from any record is included.
    """
    total = len(records)
    already = sum(1 for r in records if is_recognized_hash(r.get("password")))
    return {
        "total": total,
        "already_hashed": already,
        "candidates": total - already,
    }


def migrate(records, apply=False):
    """Return ``(new_records, summary)``. Pure; input is never mutated.

    With ``apply=False`` (default) the returned records are copies with the same
    passwords - a rehearsal. With ``apply=True`` only candidate passwords (those
    that are not already recognised hashes) are replaced with
    ``generate_password_hash``. Every other field is copied verbatim.

    Idempotent: running again on the output of an applied migration finds no
    candidates among non-empty values, because the outputs are recognised hashes.
    An empty/None password is left as-is and remains a candidate by definition.
    """
    out = []
    already = 0
    candidates = 0
    for record in records:
        new = dict(record)
        pw = record.get("password")
        if is_recognized_hash(pw):
            already += 1
        else:
            candidates += 1
            if apply and pw:
                new["password"] = generate_password_hash(pw)
        out.append(new)
    return out, {
        "total": len(records),
        "already_hashed": already,
        "candidates": candidates,
        "applied": bool(apply),
    }


def _records_from_live():
    """Yield live user records. Imported lazily so importing this module never
    opens a database connection."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import User

    for user in User.objects.all().only(
            "id", "username", "password", "is_admin", "is_active"):
        yield {
            "_id": str(user.id),
            "username": user.username,
            "password": user.password,
            "is_admin": bool(user.is_admin),
            "is_active": bool(user.is_active),
        }


def _apply_to_live(records):
    """Write the migrated passwords back. Reachable only behind the
    authorization gate in ``main``. Kept intentionally small."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import User

    written = 0
    for record in records:
        User.objects(id=record["_id"]).update_one(set__password=record["password"])
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the migration (requires authorization)")
    args = parser.parse_args(argv)

    records = list(_records_from_live())

    if not args.apply:
        summary = plan(records)
        print("DRY RUN (no writes): total={total} already_hashed={already_hashed} "
              "candidates={candidates}".format(**summary))
        return 0

    if os.environ.get(AUTHORIZATION_ENV) != AUTHORIZATION_TOKEN:
        print("Refusing to write: password migration is not authorized. "
              "A future authorization mechanism must set {} to the agreed "
              "token before --apply will proceed.".format(AUTHORIZATION_ENV),
              file=sys.stderr)
        return 2

    migrated, summary = migrate(records, apply=True)
    written = _apply_to_live(migrated)
    print("APPLIED: total={total} changed={candidates} written={written}".format(
        written=written, **summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

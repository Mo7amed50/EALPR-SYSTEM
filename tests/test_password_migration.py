"""Phase 2 / Block D Item 14B - password migration prototype tests.

Synthetic data only. No database connection, no user document, no real
credential. Every assertion runs against plain in-memory records.
"""
import io
import logging
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"),
)
import migrate_passwords as mig  # noqa: E402

# Deliberately fake, non-reusable test values.
PLAINTEXT = "synthetic-plaintext-value"
WRONG = "synthetic-wrong-value"
SHORT = "pw"          # short, but must NOT be assumed plaintext by format alone
LONG_NONHASH = "x" * 200


def fixture():
    real_hash = generate_password_hash("synthetic-already-hashed")
    return {
        "real_hash": real_hash,
        "records": [
            {"_id": "1", "username": "synthetic_a", "is_admin": True,
             "is_active": True, "password": PLAINTEXT},
            {"_id": "2", "username": "synthetic_b", "is_admin": False,
             "is_active": False, "password": real_hash},
            {"_id": "3", "username": "synthetic_c", "is_admin": False,
             "is_active": True, "password": SHORT},
            {"_id": "4", "username": "synthetic_d", "is_admin": False,
             "is_active": True, "password": LONG_NONHASH},
        ],
    }


class ClassificationTests(unittest.TestCase):
    def test_recognizes_werkzeug_hash(self):
        self.assertTrue(mig.is_recognized_hash(generate_password_hash("whatever")))

    def test_format_not_length_drives_classification(self):
        # Short is not assumed plaintext; long is not assumed a hash.
        self.assertFalse(mig.is_recognized_hash(SHORT))
        self.assertFalse(mig.is_recognized_hash(LONG_NONHASH))

    def test_colon_bearing_string_is_not_a_hash(self):
        self.assertFalse(mig.is_recognized_hash("pbkdf2:sha256:notahash"))
        self.assertFalse(mig.is_recognized_hash("user:name"))
        self.assertFalse(mig.is_recognized_hash(None))
        self.assertFalse(mig.is_recognized_hash(""))


class MigrationBehaviourTests(unittest.TestCase):
    def test_plaintext_is_hashed_and_verifies(self):
        data = fixture()
        out, summary = mig.migrate(data["records"], apply=True)
        row = out[0]
        self.assertNotEqual(row["password"], PLAINTEXT)
        self.assertTrue(mig.is_recognized_hash(row["password"]))
        self.assertTrue(check_password_hash(row["password"], PLAINTEXT))
        self.assertFalse(check_password_hash(row["password"], WRONG))
        self.assertEqual(summary["candidates"], 3)  # rows 1, 3, 4

    def test_valid_authenticates_invalid_does_not(self):
        data = fixture()
        out, _ = mig.migrate(data["records"], apply=True)
        hashed = out[2]["password"]  # SHORT row
        self.assertTrue(check_password_hash(hashed, SHORT))
        self.assertFalse(check_password_hash(hashed, SHORT + "x"))

    def test_recognized_hash_is_not_rehashed(self):
        data = fixture()
        for row in data["records"]:
            if row["_id"] == "2":
                original = row["password"]
        out, _ = mig.migrate(data["records"], apply=True)
        for row in out:
            if row["_id"] == "2":
                self.assertEqual(row["password"], original)
                self.assertTrue(check_password_hash(row["password"],
                                                     "synthetic-already-hashed"))

    def test_dry_run_makes_no_change(self):
        data = fixture()
        before = [dict(r) for r in data["records"]]
        out, summary = mig.migrate(data["records"], apply=False)
        self.assertEqual(out, before)
        self.assertFalse(summary["applied"])
        self.assertEqual(summary["candidates"], 3)

    def test_migration_is_idempotent(self):
        data = fixture()
        first, _ = mig.migrate(data["records"], apply=True)
        second, summary2 = mig.migrate(first, apply=True)
        self.assertEqual(first, second)
        # every non-empty value is now a recognised hash
        self.assertEqual(summary2["candidates"], 0)

    def test_preserved_fields_and_counts(self):
        data = fixture()
        out, _ = mig.migrate(data["records"], apply=True)
        self.assertEqual(len(out), len(data["records"]))
        for before, after in zip(data["records"], out):
            for field in ("_id", "username", "is_admin", "is_active"):
                self.assertEqual(before[field], after[field])

    def test_input_records_not_mutated(self):
        data = fixture()
        snapshot = [dict(r) for r in data["records"]]
        mig.migrate(data["records"], apply=True)
        self.assertEqual(data["records"], snapshot)


class NoLeakTests(unittest.TestCase):
    def test_migrate_prints_and_logs_no_values(self):
        data = fixture()
        buf_out, buf_err = io.StringIO(), io.StringIO()
        logs = io.StringIO()
        handler = logging.StreamHandler(logs)
        root = logging.getLogger()
        root.addHandler(handler)
        old_level = root.level
        root.setLevel(logging.DEBUG)
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                mig.plan(data["records"])
                mig.migrate(data["records"], apply=True)
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)

        blob = buf_out.getvalue() + buf_err.getvalue() + logs.getvalue()
        self.assertNotIn(PLAINTEXT, blob)
        self.assertNotIn(SHORT, blob)
        self.assertNotIn(data["real_hash"], blob)
        for row in data["records"]:
            self.assertNotIn(row["password"], blob)


class AuthorizationGateTests(unittest.TestCase):
    def test_apply_refuses_without_authorization_token(self):
        os.environ.pop(mig.AUTHORIZATION_ENV, None)
        err = io.StringIO()
        with mock.patch.object(mig, "_records_from_live", return_value=[]):
            with redirect_stderr(err):
                rc = mig.main(["--apply"])
        self.assertEqual(rc, 2)
        self.assertIn("not authorized", err.getvalue())


if __name__ == "__main__":
    import unittest.mock  # noqa: F401  (kept local to the gate test)
    unittest.main()

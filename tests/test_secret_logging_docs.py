"""Block C5 (S5, S6, S7): production secrets, startup logging, documentation.

These tests are deliberately self-contained: no real secret, password or
connection string is required anywhere, so the suite stays deterministic on a
clean checkout. Where a test needs a credential to exist it invents a sentinel
value and asserts that the value is *refused* or *never printed* - never that a
particular secret is configured.

Lifecycle note: the baseline at the time this file was added was 182 passing
tests and 2 skipped (tests/test_runtime_lifecycle.py from C5 S1/S3/S4).
"""

import logging
import os
import re
import unittest
from contextlib import contextmanager
from unittest import mock

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

import config
from config import RuntimeConfigError, assert_production_secrets

import ealpr
from ealpr import extensions

#: Every variable Block C5 owns. Snapshot/reset around each test so an
#: EALPR_ENV=production test can never leak into the next one.
_RUNTIME_VARS = (
    "EALPR_ENV",
    "EALPR_DEBUG",
    "EALPR_RELOADER",
    "EALPR_WORKERS",
)


@contextmanager
def env_vars(**values):
    """Run a block with a known, fully reset set of runtime variables.

    Copies of the real environment are restored on exit, including on failure,
    so a rejected startup cannot leave EALPR_ENV=production behind for later
    tests.
    """
    keys = set(_RUNTIME_VARS) | set(values)
    saved = {k: os.environ.get(k) for k in keys}
    for k in _RUNTIME_VARS:
        os.environ.pop(k, None)
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def log_text(captured):
    return "\n".join(r.getMessage() for r in captured.records)


# ---------------------------------------------------------------------------
# S5 - production secret and admin credential validation
# ---------------------------------------------------------------------------


class ProductionSecretValidationTests(unittest.TestCase):
    """Production must refuse placeholder credentials; everything else works."""

    VALID_SECRET = "unit-test-unique-secret-2f4c9a7e1b6d"
    VALID_ADMIN_PASSWORD = "unit-test-strong-admin-password"

    def test_production_boots_with_valid_secrets(self):
        with mock.patch.object(ealpr, "SECRET_KEY", self.VALID_SECRET), \
                mock.patch.object(ealpr, "DEFAULT_ADMIN_PASSWORD",
                                  self.VALID_ADMIN_PASSWORD), \
                env_vars(EALPR_ENV="production"):
            app = ealpr.create_app()

        self.assertEqual(app.config["EALPR_ENV"], "production")
        self.assertFalse(app.debug)
        self.assertFalse(app.config["EALPR_DEBUG"])

    def test_production_refuses_missing_secret_key(self):
        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets("production",
                                      secret_key="   ",
                                      admin_password=self.VALID_ADMIN_PASSWORD)
        self.assertIn("SECRET_KEY", str(cm.exception))

    def test_production_refuses_development_fallback_secret(self):
        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets(
                "production",
                secret_key=config.DEVELOPMENT_SECRET_FALLBACK,
                admin_password=self.VALID_ADMIN_PASSWORD)

        message = str(cm.exception)
        self.assertIn("SECRET_KEY", message)
        # The message names the problem without repeating the secret.
        self.assertNotIn(config.DEVELOPMENT_SECRET_FALLBACK, message)

    def test_production_refuses_template_placeholder_secret(self):
        placeholder = "your-super-secret-key-change-this"
        self.assertIn(placeholder, config.PLACEHOLDER_SECRET_KEYS)

        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets("production",
                                      secret_key=placeholder,
                                      admin_password=self.VALID_ADMIN_PASSWORD)
        self.assertNotIn(placeholder, str(cm.exception))

    def test_production_refuses_missing_admin_password(self):
        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets("production",
                                      secret_key=self.VALID_SECRET,
                                      admin_password="")
        self.assertIn("DEFAULT_ADMIN_PASSWORD", str(cm.exception))

    def test_production_refuses_known_default_admin_password(self):
        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets(
                "production",
                secret_key=self.VALID_SECRET,
                admin_password=config.KNOWN_DEFAULT_ADMIN_PASSWORD)

        message = str(cm.exception)
        self.assertIn("DEFAULT_ADMIN_PASSWORD", message)
        self.assertNotIn(config.KNOWN_DEFAULT_ADMIN_PASSWORD, message)

    def test_error_message_never_echoes_the_values(self):
        # A perfectly valid value must not be echoed just because some other
        # field is the one that failed.
        valid_secret = "ZZZ-CANARY-SECRET-1f8e3d"
        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets("production",
                                      secret_key=valid_secret,
                                      admin_password="")
        self.assertNotIn(valid_secret, str(cm.exception))

        # And refused values must not be echoed either, even when several
        # problems are reported in one message.
        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets(
                "production",
                secret_key=config.DEVELOPMENT_SECRET_FALLBACK,
                admin_password=config.KNOWN_DEFAULT_ADMIN_PASSWORD)

        message = str(cm.exception)
        self.assertNotIn(config.DEVELOPMENT_SECRET_FALLBACK, message)
        self.assertNotIn(config.KNOWN_DEFAULT_ADMIN_PASSWORD, message)

    def test_multiple_problems_are_reported_together(self):
        with self.assertRaises(RuntimeConfigError) as cm:
            assert_production_secrets("production",
                                      secret_key="",
                                      admin_password="")
        message = str(cm.exception)
        self.assertIn("SECRET_KEY", message)
        self.assertIn("DEFAULT_ADMIN_PASSWORD", message)

    def test_development_still_accepts_the_fallbacks(self):
        # Regression guard for the zero-setup local workflow: development and
        # test must keep working with a fresh clone and no .env at all.
        assert_production_secrets("development",
                                  secret_key=config.DEVELOPMENT_SECRET_FALLBACK,
                                  admin_password=config.KNOWN_DEFAULT_ADMIN_PASSWORD)
        assert_production_secrets("development", secret_key="", admin_password="")

    def test_test_mode_is_deterministic_and_needs_no_secrets(self):
        first = assert_production_secrets("test")
        second = assert_production_secrets("test")
        self.assertEqual(first, second)
        self.assertIsNone(first)

    def test_create_app_refuses_production_with_default_secrets(self):
        # Boot-level check: the rejection happens in the factory, before any
        # Flask object or database connection exists.
        with mock.patch.object(ealpr, "SECRET_KEY",
                               config.DEVELOPMENT_SECRET_FALLBACK), \
                mock.patch.object(ealpr, "DEFAULT_ADMIN_PASSWORD",
                                  config.KNOWN_DEFAULT_ADMIN_PASSWORD), \
                env_vars(EALPR_ENV="production"):
            with self.assertRaises(RuntimeConfigError) as cm:
                ealpr.create_app()

        message = str(cm.exception)
        self.assertIn("Production startup refused", message)
        self.assertNotIn(config.DEVELOPMENT_SECRET_FALLBACK, message)
        self.assertNotIn(config.KNOWN_DEFAULT_ADMIN_PASSWORD, message)

    def test_create_app_is_unaffected_outside_production(self):
        with mock.patch.object(ealpr, "SECRET_KEY",
                               config.DEVELOPMENT_SECRET_FALLBACK), \
                mock.patch.object(ealpr, "DEFAULT_ADMIN_PASSWORD",
                                  config.KNOWN_DEFAULT_ADMIN_PASSWORD), \
                env_vars(EALPR_ENV="development"):
            app = ealpr.create_app()
        self.assertEqual(app.config["EALPR_ENV"], "development")


# ---------------------------------------------------------------------------
# S6 - startup logging discipline
# ---------------------------------------------------------------------------


class StartupLoggingTests(unittest.TestCase):
    """Startup reports through logging, and never through a credential."""

    def test_startup_sources_use_logging_not_print(self):
        for name in ("config.py", os.path.join("ealpr", "__init__.py")):
            with open(os.path.join(BASE_DIR, name), encoding="utf-8") as handle:
                source = handle.read()
            self.assertIsNone(
                re.search(r"\bprint\s*\(", source),
                "{} still calls print(); startup must use logging".format(name))

    def test_create_app_emits_startup_logs(self):
        with self.assertLogs("ealpr", level="INFO") as captured:
            app = ealpr.create_app()

        message = log_text(captured)
        self.assertIn("Starting EALPR in", message)
        self.assertEqual(app.config["EALPR_ENV"], "development")

    def test_startup_logs_contain_no_sensitive_values(self):
        sentinel_secret = "ZZZ-LOG-SECRET-4e19c2ab"
        sentinel_admin = "ZZZ-LOG-ADMIN-PASSWORD-93f1"

        with mock.patch.object(ealpr, "SECRET_KEY", sentinel_secret), \
                mock.patch.object(ealpr, "DEFAULT_ADMIN_PASSWORD",
                                  sentinel_admin), \
                self.assertLogs("ealpr", level="INFO") as captured:
            ealpr.create_app()

        message = log_text(captured)
        self.assertNotIn(sentinel_secret, message)
        self.assertNotIn(sentinel_admin, message)

    def test_redact_credentials_strips_userinfo_but_keeps_host(self):
        redacted = ealpr._redact_credentials(
            "mongodb+srv://alice:pw123@cluster1.example.mongodb.net/ealpr_db")

        self.assertNotIn("alice:pw123", redacted)
        self.assertNotIn("pw123", redacted)
        # Operationally useful context survives: host, path and driver text.
        self.assertIn("cluster1.example", redacted)
        self.assertIn("ealpr_db", redacted)

    def test_redact_credentials_handles_exception_text(self):
        error = RuntimeError("auth failed for mongodb://bob:topsecret@db.local/x")

        redacted = ealpr._redact_credentials(error)

        self.assertNotIn("topsecret", redacted)
        self.assertIn("auth failed", redacted)

    def test_invalid_mongodb_uri_credentials_are_never_logged(self):
        original_uri = ealpr.MONGODB_URI
        original_flag = extensions.DB_ENABLED
        sentinel_uri = "hunter2:sup3rs3cr3t@not-a-real-host.invalid/db"

        try:
            ealpr.MONGODB_URI = sentinel_uri
            with self.assertLogs("ealpr", level="WARNING") as captured:
                connected = ealpr._connect_mongodb()
        finally:
            ealpr.MONGODB_URI = original_uri
            extensions.DB_ENABLED = original_flag

        self.assertFalse(connected)

        message = log_text(captured)
        self.assertNotIn("hunter2", message)
        self.assertNotIn("sup3rs3cr3t", message)
        # The failure itself is still reported, so operators can act on it.
        self.assertIn("scheme", message)

    def test_startup_logging_handler_is_installed_at_most_once(self):
        ealpr.create_app()
        first = len(ealpr.log.handlers)
        ealpr.create_app()

        self.assertLessEqual(len(ealpr.log.handlers), 1)
        self.assertEqual(first, len(ealpr.log.handlers))


# ---------------------------------------------------------------------------
# S7 - runtime and deployment documentation
# ---------------------------------------------------------------------------


class DocumentationTests(unittest.TestCase):
    """The docs must state the real configuration story, not an aspiration."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8") as handle:
            cls.readme = handle.read()
        with open(os.path.join(BASE_DIR, ".env.example"),
                  encoding="utf-8") as handle:
            cls.env_example = handle.read()

    RUNTIME_VARIABLES = ("EALPR_ENV", "EALPR_DEBUG", "EALPR_RELOADER",
                         "EALPR_WORKERS")

    def test_documents_contain_all_four_runtime_variables(self):
        for variable in self.RUNTIME_VARIABLES:
            self.assertIn(variable, self.readme,
                          "README.md does not document {}".format(variable))
            self.assertIn(variable, self.env_example,
                          ".env.example does not document {}".format(variable))

    def test_documents_state_the_single_process_requirement(self):
        self.assertIn("single process", self.readme)

    def test_documents_state_the_debug_restriction(self):
        self.assertIn("EALPR_ENV=development", self.readme)
        self.assertIn("RuntimeConfigError", self.readme)

    def test_documents_state_the_production_secret_requirements(self):
        for document, name in ((self.readme, "README.md"),
                               (self.env_example, ".env.example")):
            self.assertIn("SECRET_KEY", document)
            self.assertIn("DEFAULT_ADMIN_PASSWORD", document)

        # The two refused development values are named so operators recognise
        # them, without any real credential appearing anywhere.
        self.assertIn(config.DEVELOPMENT_SECRET_FALLBACK, self.readme)
        self.assertIn(config.KNOWN_DEFAULT_ADMIN_PASSWORD, self.readme)

    def test_documents_state_the_startup_command(self):
        self.assertIn("python app.py", self.readme)

    def test_documents_state_that_s2_is_implemented(self):
        self.assertIn("S2", self.readme)
        self.assertIn("implemented", self.readme)

    def test_mentions_production_readiness_limitations(self):
        # After implementing S2, we no longer claim "not production-ready"
        # as an absolute; instead we document remaining limitations.
        self.assertIn("not fully production-ready", self.readme)

    def test_wsgi_server_is_documented(self):
        forbidden_other = ("gunicorn", "uvicorn", "hypercorn",
                           "gevent", "eventlet")
        for document, name in ((self.readme, "README.md"),
                                (self.env_example, ".env.example")):
            lowered = document.lower()
            # Must document waitress
            self.assertIn("waitress", lowered,
                          "{} must document waitress".format(name))
            for server in forbidden_other:
                self.assertNotIn(
                    server, lowered,
                    "{} mentions {} which was not selected".format(
                        name, server))

    def test_env_example_ships_only_placeholders(self):
        secret = re.search(r"^SECRET_KEY=(.*)$", self.env_example, re.MULTILINE)
        self.assertIsNotNone(secret)
        self.assertIn(secret.group(1).strip(), config.PLACEHOLDER_SECRET_KEYS)

        admin = re.search(r"^DEFAULT_ADMIN_PASSWORD=(.*)$", self.env_example,
                          re.MULTILINE)
        self.assertIsNotNone(admin)
        self.assertEqual(admin.group(1).strip(), "")

        for line in self.env_example.splitlines():
            if line.startswith("MONGODB_URI="):
                self.assertIn("<", line,
                              "MONGODB_URI in .env.example is not a placeholder")


if __name__ == "__main__":
    unittest.main()

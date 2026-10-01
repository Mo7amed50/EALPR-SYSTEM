"""Block C5 (S1/S3/S4) — runtime mode, single-process policy, exit shutdown.

Read-only with respect to the database: nothing here writes to MongoDB, and no
worker is started. The S4 tests use a stand-in module so that the hook's wiring
is verified without disturbing the real inference worker that the rest of the
suite relies on.
"""

import os
import sys
import unittest
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BASE_DIR, ".env"))

import config  # noqa: E402
from config import RuntimeConfigError  # noqa: E402

#: Every variable the C5 switches read. Cleared around each test so a value set
#: by one case cannot leak into the next, and so the developer's own shell or
#: .env never changes the expected result.
C5_VARS = ("EALPR_ENV", "EALPR_DEBUG", "EALPR_RELOADER", "EALPR_WORKERS")


@contextmanager
def runtime_env(**values):
    """Run with exactly the given C5 environment, restoring it afterwards."""
    saved = {name: os.environ.get(name) for name in C5_VARS}
    try:
        for name in C5_VARS:
            os.environ.pop(name, None)
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class RuntimeModeTests(unittest.TestCase):
    """S1 — EALPR_ENV / EALPR_DEBUG resolution."""

    def test_defaults_to_development_without_debug(self):
        """With nothing set the app is usable in development and has no debugger.

        This is the behaviour that replaces the hard-coded `debug=True`: the
        common case no longer exposes the interactive debugger, and a developer
        still gets a working server.
        """
        with runtime_env():
            self.assertEqual(config.resolve_runtime_mode(), ("development", False))

    def test_development_may_enable_debug(self):
        """Debug stays available for development, but only when asked for."""
        with runtime_env(EALPR_ENV="development", EALPR_DEBUG="1"):
            self.assertEqual(config.resolve_runtime_mode(), ("development", True))

    def test_production_defaults_to_debug_off(self):
        with runtime_env(EALPR_ENV="production"):
            self.assertEqual(config.resolve_runtime_mode(), ("production", False))

    def test_production_rejects_debug(self):
        """C5.2: production + debug must fail loudly, not be downgraded."""
        for value in ("1", "true", "yes", "on", "TRUE", "On"):
            with self.subTest(debug=value):
                with runtime_env(EALPR_ENV="production", EALPR_DEBUG=value):
                    with self.assertRaises(RuntimeConfigError) as ctx:
                        config.resolve_runtime_mode()
        # The message must name both settings and say why, so the operator can
        # act on it without reading the source.
        message = str(ctx.exception)
        self.assertIn("EALPR_DEBUG", message)
        self.assertIn("production", message)

    def test_explicit_production_debug_off_is_accepted(self):
        with runtime_env(EALPR_ENV="production", EALPR_DEBUG="0"):
            self.assertEqual(config.resolve_runtime_mode(), ("production", False))

    def test_test_environment_is_accepted_and_deterministic(self):
        """S1: `test` must resolve the same way every time it is asked for."""
        with runtime_env(EALPR_ENV="test", EALPR_DEBUG="0"):
            results = [config.resolve_runtime_mode() for _ in range(3)]
            self.assertEqual(results, [("test", False)] * 3)

    def test_test_environment_rejects_debug(self):
        """Test runs stay deterministic: the debugger can hang a failing run.

        A request that cannot be honoured raises here, the same as production,
        rather than being quietly dropped.
        """
        with runtime_env(EALPR_ENV="test", EALPR_DEBUG="1"):
            with self.assertRaises(RuntimeConfigError):
                config.resolve_runtime_mode()

    def test_invalid_environment_is_rejected(self):
        """An unknown mode must not silently fall back to a default."""
        for value in ("staging", "prod", "dev", "deploy", "PRODUCTIONX"):
            with self.subTest(env=value):
                with runtime_env(EALPR_ENV=value):
                    with self.assertRaises(RuntimeConfigError) as ctx:
                        config.resolve_runtime_mode()
                self.assertIn("EALPR_ENV", str(ctx.exception))

    def test_environment_tolerates_surrounding_whitespace_and_case(self):
        """A stray space or upper case is a typo, not a different intention.

        Only whitespace and case are normalised. Anything else is rejected, so
        this tolerance cannot be used to reach a mode that was not requested.
        """
        for value in ("PRODUCTION", "production", "  production  "):
            with self.subTest(env=value):
                with runtime_env(EALPR_ENV=value):
                    self.assertEqual(config.resolve_runtime_mode(), ("production", False))

    def test_invalid_boolean_is_rejected(self):
        """An ambiguous flag is a configuration error, not a guess."""
        for value in ("maybe", "2", "y es"):
            with self.subTest(debug=value):
                with runtime_env(EALPR_DEBUG=value):
                    with self.assertRaises(RuntimeConfigError):
                        config.resolve_runtime_mode()

    def test_all_documented_environments_are_accepted(self):
        for name in config.VALID_ENVIRONMENTS:
            with self.subTest(env=name):
                with runtime_env(EALPR_ENV=name):
                    resolved, _ = config.resolve_runtime_mode()
                    self.assertEqual(resolved, name)


class SingleProcessTests(unittest.TestCase):
    """S3 — the C2/C3/C4 single-process constraint, enforced rather than noted."""

    def test_default_configuration_is_single_process(self):
        with runtime_env():
            config.assert_single_process()  # must not raise

    def test_single_worker_explicitly_allowed(self):
        with runtime_env(EALPR_WORKERS="1"):
            config.assert_single_process()

    def test_multiple_workers_rejected(self):
        """C5.3: a second process must fail loudly rather than run unsafely."""
        for value in ("2", "4", "16"):
            with self.subTest(workers=value):
                with runtime_env(EALPR_WORKERS=value):
                    with self.assertRaises(RuntimeConfigError) as ctx:
                        config.assert_single_process()
                self.assertIn("single process", str(ctx.exception))

    def test_reloader_rejected(self):
        """The reloader is the near-miss that starts a hidden second process."""
        for value in ("1", "true", "on"):
            with self.subTest(reloader=value):
                with runtime_env(EALPR_RELOADER=value):
                    with self.assertRaises(RuntimeConfigError) as ctx:
                        config.assert_single_process()
                self.assertIn("second process", str(ctx.exception))

    def test_reloader_explicitly_off_is_allowed(self):
        with runtime_env(EALPR_RELOADER="false"):
            config.assert_single_process()

    def test_non_integer_workers_rejected(self):
        with runtime_env(EALPR_WORKERS="many"):
            with self.assertRaises(RuntimeConfigError):
                config.assert_single_process()

    def test_production_does_not_relax_the_single_process_rule(self):
        """The policy is not environment-dependent: it is a design constraint."""
        with runtime_env(EALPR_ENV="production", EALPR_WORKERS="8"):
            with self.assertRaises(RuntimeConfigError):
                config.assert_single_process()


class AppFactoryRuntimeTests(unittest.TestCase):
    """S1/S3 — the factory surfaces the switches, and honours the rejections."""

    def test_create_app_records_runtime_mode(self):
        """The booted app reports the mode it resolved, for diagnostics."""
        from ealpr import create_app

        with runtime_env(EALPR_ENV="development"):
            app = create_app()
            self.assertEqual(app.config["EALPR_ENV"], "development")
            self.assertFalse(app.debug)

    def test_create_app_rejects_invalid_configuration(self):
        """An unusable configuration must stop startup, before the app exists."""
        from ealpr import create_app

        with runtime_env(EALPR_ENV="production", EALPR_DEBUG="1"):
            with self.assertRaises(RuntimeConfigError):
                create_app()
        with runtime_env(EALPR_WORKERS="2"):
            with self.assertRaises(RuntimeConfigError):
                create_app()


class AtexitShutdownTests(unittest.TestCase):
    """S4 — the existing `shutdown()` is wired to interpreter exit, once."""

    def test_hook_registration_is_idempotent(self):
        """A second create_app() must not stack a second hook."""
        from ealpr.inference_jobs import (
            register_atexit_shutdown,
            atexit_shutdown_registered,
            _atexit_registered,
        )
        import ealpr.inference_jobs as ij

        original = ij._atexit_registered
        try:
            ij._atexit_registered = False
            self.assertFalse(atexit_shutdown_registered())
            self.assertTrue(register_atexit_shutdown())
            self.assertTrue(atexit_shutdown_registered())
            # Second and third calls are no-ops.
            self.assertFalse(register_atexit_shutdown())
            self.assertFalse(register_atexit_shutdown())
            self.assertTrue(atexit_shutdown_registered())
        finally:
            ij._atexit_registered = original

    def test_registered_callback_invokes_shutdown(self):
        """The hook must call the real shutdown(), not a re-implementation."""
        import ealpr.inference_jobs as ij

        calls = []
        original_shutdown = ij.shutdown
        original_registered = ij._atexit_registered
        try:
            ij.shutdown = lambda *a, **k: calls.append(1)
            ij._atexit_registered = False

            self.assertTrue(ij.register_atexit_shutdown())

            # Invoke the module's callback directly. Running the real atexit
            # stack would also fire every other module's hook.
            ij._atexit_shutdown()
            self.assertEqual(len(calls), 1)
        finally:
            ij.shutdown = original_shutdown
            ij._atexit_registered = original_registered

    def test_shutdown_callback_never_raises(self):
        """At interpreter exit a raise would mask the real exit status."""
        import ealpr.inference_jobs as ij

        original_shutdown = ij.shutdown

        def exploding_shutdown(*args, **kwargs):
            raise RuntimeError("simulated failure during interpreter teardown")

        try:
            ij.shutdown = exploding_shutdown
            ij._atexit_shutdown()  # must not raise
        finally:
            ij.shutdown = original_shutdown

    def test_shutdown_does_not_mutate_database_collections(self):
        """C5.5: shutdown only touches threads and the in-memory queue.

        It must not write to `detection_result` or `visitor`, so the exit path
        cannot be the thing that changes stored data.
        """
        from ealpr import create_app
        from models import DetectionResult, Visitor

        create_app()  # ensure the collection handles are usable
        before_det = DetectionResult.objects.count()
        before_vis = Visitor.objects.count()

        from ealpr.inference_jobs import shutdown
        shutdown()

        self.assertEqual(DetectionResult.objects.count(), before_det)
        self.assertEqual(Visitor.objects.count(), before_vis)

    def test_shutdown_is_safe_to_call_repeatedly(self):
        """C5.5: repeated calls are safe, matching C4's documented behaviour."""
        from ealpr.inference_jobs import shutdown

        # shutdown() returns a bool and tolerates repeat calls; neither may raise.
        for _ in range(2):
            self.assertIsNotNone(shutdown())


if __name__ == "__main__":
    unittest.main(verbosity=2)
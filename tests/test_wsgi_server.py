"""Block C5 (S2) — production WSGI server (Waitress) implementation tests.

S2 adds a production WSGI server without changing the single-process,
threaded-Socket.IO architecture. These tests cover the implementation itself:

* the dependency is pinned and no broker / multi-worker package was added;
* the documentation states the exact development and production commands,
  the loopback default, an explicit thread count, the long-polling transport,
  the absence of WebSocket under Waitress, and does not claim full production
  readiness;
* the single-process / reloader / production-secret guards are still intact;
* importing the WSGI entry point initialises the application exactly once;
* a real Waitress subprocess boots the app in production mode and serves HTTP;
* against a real running Waitress process, Socket.IO negotiates long-polling,
  the WebSocket upgrade fails server-side, and a ``live_detection`` event
  emitted from a background thread through the existing ``socketio.emit``
  mechanism is delivered to a connected client.

Nothing here writes to MongoDB. The production smoke test uses ephemeral
test-only secrets that are never persisted or printed.
"""

import http.client
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BASE_DIR, ".env"))

import config  # noqa: E402

REQUIREMENTS_PATH = os.path.join(BASE_DIR, "requirements.txt")
README_PATH = os.path.join(BASE_DIR, "README.md")
ENV_EXAMPLE_PATH = os.path.join(BASE_DIR, ".env.example")

try:  # pragma: no cover - presence is asserted separately
    import waitress  # noqa: F401
    HAS_WAITRESS = True
except Exception:  # pragma: no cover
    HAS_WAITRESS = False

try:
    import socketio as sio_client
    HAS_SIO_CLIENT = True
except Exception:  # pragma: no cover
    HAS_SIO_CLIENT = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def pinned_waitress_version():
    match = re.search(r"^waitress==(.+?)\s*$", read(REQUIREMENTS_PATH),
                      re.MULTILINE)
    return match.group(1) if match else None


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def child_env(extra=None):
    """A clean EALPR environment for a server subprocess.

    Inherits the real environment (so MongoDB credentials from ``.env`` are
    available the same way the app normally sees them) but scrubs and re-sets
    the Block C5 switches so a developer's shell cannot change the outcome.
    """
    env = dict(os.environ)
    for name in ("EALPR_ENV", "EALPR_DEBUG", "EALPR_RELOADER", "EALPR_WORKERS"):
        env.pop(name, None)
    env["EALPR_ENV"] = "development"
    env["PYTHONPATH"] = BASE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    if extra:
        env.update(extra)
    return env


class ServerProcess:
    """A subprocess running a waitress-served application, with teardown."""

    def __init__(self, script, threads, env, port):
        self.port = port
        self._output = None
        self.proc = subprocess.Popen(
            [sys.executable, script, str(port), str(threads)],
            cwd=BASE_DIR,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def wait_ready(self, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    "server exited before becoming ready:\n" + self.stop())
            try:
                with socket.create_connection(("127.0.0.1", self.port),
                                              timeout=1):
                    return True
            except OSError:
                time.sleep(0.3)
        raise AssertionError(
            "server did not become ready on port {}:\n{}".format(
                self.port, self.stop()))

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def stop(self):
        """Terminate the server and return its combined output (idempotent)."""
        if self._output is not None:
            return self._output
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self._output = self.proc.communicate(timeout=20)[0] or ""
        except Exception:
            self.proc.kill()
            self._output = self.proc.communicate(timeout=20)[0] or ""
        return self._output


# Driver scripts are written outside the repository so they can never be
# committed or mistaken for application code.
_EMIT_DRIVER = textwrap.dedent("""
    import sys
    import threading

    from waitress import serve

    port = int(sys.argv[1])
    threads = int(sys.argv[2])

    import app as app_module
    from ealpr.extensions import socketio

    def control():
        for line in sys.stdin:
            command = line.strip()
            if command == "EMIT":
                # The same call the detection producers make from their own
                # background threads (live_detection is their only event).
                socketio.emit("live_detection",
                              {"plate_number": "__s2_integration__", "seq": 1})
                sys.stdout.write("EMITTED\\n")
                sys.stdout.flush()
            elif command == "QUIT":
                break

    threading.Thread(target=control, daemon=True).start()
    sys.stdout.write("DRIVER_READY\\n")
    sys.stdout.flush()
    serve(app_module.app, host="127.0.0.1", port=port, threads=threads)
""")

_PLAIN_DRIVER = textwrap.dedent("""
    import sys

    from waitress import serve

    port = int(sys.argv[1])
    threads = int(sys.argv[2])

    import app as app_module

    sys.stdout.write("DRIVER_READY\\n")
    sys.stdout.flush()
    serve(app_module.app, host="127.0.0.1", port=port, threads=threads)
""")


def _write_script(source):
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8")
    handle.write(source)
    handle.close()
    return handle.name


# ---------------------------------------------------------------------------
# dependency
# ---------------------------------------------------------------------------


class WaitressDependencyTests(unittest.TestCase):
    """S2 is only real if the server is actually a pinned dependency."""

    def test_waitress_is_pinned_in_requirements(self):
        version = pinned_waitress_version()
        self.assertIsNotNone(
            version,
            "requirements.txt must pin waitress with an exact version")

    def test_pinned_version_is_installed(self):
        version = pinned_waitress_version()
        if version is None:
            self.skipTest("waitress is not pinned")
        try:
            import importlib.metadata
            installed = importlib.metadata.version("waitress")
        except Exception:
            self.skipTest("waitress is not installed in this environment")
        self.assertEqual(installed, version)

    def test_no_broker_or_multiworker_dependency_added(self):
        lowered = read(REQUIREMENTS_PATH).lower()
        for package in ("redis", "celery", "rq==", "kafka", "pika",
                        "rabbitmq", "gunicorn", "uvicorn", "hypercorn",
                        "gevent", "eventlet"):
            self.assertNotIn(
                package, lowered,
                "requirements.txt must not add {}".format(package))


# ---------------------------------------------------------------------------
# documentation
# ---------------------------------------------------------------------------


class ProductionDocumentationTests(unittest.TestCase):
    """The docs must state the real configuration story."""

    @classmethod
    def setUpClass(cls):
        cls.readme = read(README_PATH)
        cls.env_example = read(ENV_EXAMPLE_PATH)

    def test_development_command_is_documented(self):
        self.assertIn("python app.py", self.readme)

    def test_production_command_uses_app_app(self):
        for document, name in ((self.readme, "README.md"),
                               (self.env_example, ".env.example")):
            self.assertIn("waitress-serve", document,
                          "{} does not document waitress-serve".format(name))
            self.assertIn("app:app", document,
                          "{} does not use the app:app WSGI target".format(name))

    def test_production_command_binds_loopback_by_default(self):
        self.assertIn("127.0.0.1", self.readme)
        self.assertIn("127.0.0.1", self.env_example)

    def test_production_command_sets_an_explicit_thread_count(self):
        matches = re.findall(r"--threads=(\d+)", self.readme)
        self.assertTrue(matches, "README.md does not set --threads explicitly")
        self.assertTrue(all(int(value) >= 1 for value in matches))

    def test_documents_describe_long_polling(self):
        self.assertIn("long-polling", self.readme.lower())

    def test_documents_state_websocket_is_unavailable(self):
        lowered = self.readme.lower()
        self.assertIn("websocket", lowered)
        self.assertIn("not available", lowered)

    def test_documents_process_supervision_requirement(self):
        self.assertIn("supervisor", self.readme.lower())

    def test_documents_do_not_claim_full_production_readiness(self):
        self.assertIn("not fully production-ready", self.readme)

    def test_development_entry_point_remains_in_app_py(self):
        app_source = read(os.path.join(BASE_DIR, "app.py"))
        self.assertIn('if __name__ == "__main__"', app_source)
        self.assertIn("socketio.run", app_source)


# ---------------------------------------------------------------------------
# guards unchanged
# ---------------------------------------------------------------------------


class GuardRegressionTests(unittest.TestCase):
    """Implementing S2 must not weaken the C5 S1/S3/S5 guarantees."""

    def test_socketio_async_mode_is_still_threading(self):
        source = read(os.path.join(BASE_DIR, "ealpr", "extensions.py"))
        self.assertIn('async_mode="threading"', source)

    def test_default_configuration_is_single_process(self):
        saved = {k: os.environ.get(k) for k in
                 ("EALPR_RELOADER", "EALPR_WORKERS")}
        try:
            os.environ.pop("EALPR_RELOADER", None)
            os.environ.pop("EALPR_WORKERS", None)
            config.assert_single_process()  # must not raise
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_multiple_workers_still_rejected(self):
        saved = os.environ.get("EALPR_WORKERS")
        try:
            os.environ["EALPR_WORKERS"] = "2"
            with self.assertRaises(config.RuntimeConfigError):
                config.assert_single_process()
        finally:
            if saved is None:
                os.environ.pop("EALPR_WORKERS", None)
            else:
                os.environ["EALPR_WORKERS"] = saved

    def test_reloader_still_rejected(self):
        saved = os.environ.get("EALPR_RELOADER")
        try:
            os.environ["EALPR_RELOADER"] = "true"
            with self.assertRaises(config.RuntimeConfigError):
                config.assert_single_process()
        finally:
            if saved is None:
                os.environ.pop("EALPR_RELOADER", None)
            else:
                os.environ["EALPR_RELOADER"] = saved

    def test_production_secret_validation_still_refuses_defaults(self):
        with self.assertRaises(config.RuntimeConfigError):
            config.assert_production_secrets(
                "production",
                secret_key=config.DEVELOPMENT_SECRET_FALLBACK,
                admin_password=config.KNOWN_DEFAULT_ADMIN_PASSWORD)


# ---------------------------------------------------------------------------
# entry point initialises exactly once
# ---------------------------------------------------------------------------


class EntryPointInitializationTests(unittest.TestCase):
    """The WSGI target ``app:app`` must not build the app more than once."""

    def test_importing_entry_point_initializes_once(self):
        script = textwrap.dedent("""
            import ealpr

            calls = []
            _original = ealpr.create_app

            def counting(*args, **kwargs):
                calls.append(1)
                return _original(*args, **kwargs)

            ealpr.create_app = counting

            import app
            import app  # duplicate import must be served from cache

            print("CALLS", len(calls))
            print("HAS_APP", hasattr(app, "app"))
        """)
        path = _write_script(script)
        try:
            result = subprocess.run(
                [sys.executable, path],
                cwd=BASE_DIR, env=child_env(), capture_output=True, text=True,
                timeout=180,
            )
        finally:
            os.remove(path)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CALLS 1", result.stdout,
                      "entry point initialised the app more than once")
        self.assertIn("HAS_APP True", result.stdout)


# ---------------------------------------------------------------------------
# production subprocess smoke test
# ---------------------------------------------------------------------------


class WaitressSmokeTests(unittest.TestCase):
    """Boot the real app under Waitress in production mode, using test secrets."""

    def setUp(self):
        if not HAS_WAITRESS:
            self.skipTest("waitress is not installed")
        self.port = free_port()
        self.script = _write_script(_PLAIN_DRIVER)
        # Ephemeral, test-only credentials. Generated per run and never
        # persisted: they exist only in this subprocess's environment.
        self.env = child_env({
            "EALPR_ENV": "production",
            "EALPR_DEBUG": "false",
            "EALPR_RELOADER": "false",
            "EALPR_WORKERS": "1",
            "SECRET_KEY": "s2-smoke-" + secrets.token_hex(16),
            "DEFAULT_ADMIN_PASSWORD": "s2-smoke-" + secrets.token_hex(12),
        })
        self.server = ServerProcess(self.script, 12, self.env, self.port)

    def tearDown(self):
        try:
            self.server.stop()
        finally:
            if os.path.exists(self.script):
                os.remove(self.script)

    def test_serves_http_in_production_mode(self):
        self.server.wait_ready(timeout=90)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/")
        response = conn.getresponse()
        status = response.status
        response.read()
        conn.close()
        self.assertIn(status, (200, 302))

        output = self.server.stop()
        self.assertEqual(
            output.count("Starting EALPR in production mode"), 1,
            "startup ran more than once:\n" + output)


# ---------------------------------------------------------------------------
# Socket.IO integration
# ---------------------------------------------------------------------------


class SocketIOIntegrationTests(unittest.TestCase):
    """Verify the real transport behaviour against a running Waitress server."""

    def setUp(self):
        if not HAS_WAITRESS:
            self.skipTest("waitress is not installed")
        if not HAS_SIO_CLIENT:
            self.skipTest("python-socketio client is not installed")
        self.port = free_port()
        self.script = _write_script(_EMIT_DRIVER)
        self.server = ServerProcess(
            self.script, 12, child_env(), self.port)
        self.server.wait_ready(timeout=90)

    def tearDown(self):
        try:
            self.server.stop()
        finally:
            if os.path.exists(self.script):
                os.remove(self.script)

    def _handshake(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/socket.io/?EIO=4&transport=polling&t=s2")
        response = conn.getresponse()
        body = response.read().decode("utf-8", "replace")
        conn.close()
        return response.status, body

    def _attempt_websocket_upgrade(self, sid):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest(
            "GET", "/socket.io/?EIO=4&transport=websocket&sid=" + sid)
        conn.putheader("Connection", "Upgrade")
        conn.putheader("Upgrade", "websocket")
        conn.putheader("Sec-WebSocket-Version", "13")
        conn.putheader("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
        conn.endheaders()
        response = conn.getresponse()
        status = response.status
        response.read()
        conn.close()
        return status

    def test_polling_handshake_succeeds(self):
        status, body = self._handshake()
        self.assertEqual(status, 200)
        self.assertIn('"sid"', body)

    def test_websocket_upgrade_is_rejected_server_side(self):
        _status, body = self._handshake()
        match = re.search(r'"sid":"([^"]+)"', body)
        self.assertIsNotNone(match, "no session id in handshake: " + body)
        status = self._attempt_websocket_upgrade(match.group(1))
        self.assertEqual(
            status, 500,
            "expected the WebSocket upgrade to fail under waitress")

    def test_client_stays_on_long_polling_and_receives_live_detection(self):
        received = []
        client = sio_client.Client(reconnection=False)
        client.on("live_detection", lambda data: received.append(data))
        try:
            # Default transports: the client attempts an upgrade, which cannot
            # succeed under waitress, so it must remain on polling.
            client.connect("http://127.0.0.1:{}".format(self.port),
                           wait_timeout=15)
            time.sleep(2)
            self.assertEqual(
                client.transport(), "polling",
                "client did not remain on the polling transport")

            # Trigger the server to emit, from a background thread, through the
            # same socketio.emit("live_detection", ...) call the producers use.
            self.server.send("EMIT")
            deadline = time.time() + 15
            while time.time() < deadline and not received:
                time.sleep(0.2)

            self.assertTrue(
                received,
                "live_detection was not delivered over long-polling")
            self.assertEqual(received[0].get("plate_number"),
                             "__s2_integration__")
        finally:
            try:
                client.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)

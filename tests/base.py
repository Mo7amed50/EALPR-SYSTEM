"""Shared read-only test setup.

Imports must happen after ``.env`` is loaded, because ``config.py`` reads
``MONGODB_URI`` from the environment and the app factory connects eagerly.
"""

import os
import sys
import unittest
from collections import Counter
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BASE_DIR, ".env"))

from pymongo import monitoring  # noqa: E402

#: The application is built once, which is what connects MongoEngine. Building
#: it does not start the web server, the camera or any AI worker, so it is safe
#: in a test process.
_app = None


def get_app():
    global _app
    if _app is None:
        from ealpr import create_app

        _app = create_app()
    return _app


class CommandSpy(monitoring.CommandListener):
    """Records the MongoDB commands a block of code actually issues.

    This is what makes the "no N+1" and "images are never listed" claims in the
    C1 report checkable. If a route started querying once per row, the
    ``find``/``aggregate`` count would scale with the number of documents; if a
    list query started pulling image payloads, ``projections`` would contain
    the binary field names.

    pymongo offers no way to unregister a listener, so a single instance is
    registered for the life of the process and recording is gated by
    :attr:`enabled`.
    """

    def __init__(self):
        self.enabled = False
        self.commands = Counter()
        self.projections = []
        super().__init__()

    def started(self, event):
        if not self.enabled:
            return
        # Counted here only. A command produces both a started and a succeeded
        # event, so counting both would double everything. ``started`` is also
        # the only one that exposes the raw command, which is where the
        # projection lives.
        self.commands[event.command_name] += 1
        if event.command_name in ("find", "aggregate"):
            self.projections.append(dict(event.command.get("projection") or {}))

    def succeeded(self, event):
        # Nothing to do; see started().
        return

    def capture(self):
        """Context manager that records only inside the ``with`` body."""
        return _Capture(self)

    @property
    def reads(self):
        """Commands that open a cursor: a per-row query would grow this."""
        return sum(
            self.commands.get(name, 0)
            for name in ("find", "aggregate", "count", "distinct")
        )


class _Capture:
    def __init__(self, spy):
        self.spy = spy

    def __enter__(self):
        self.spy.commands.clear()
        self.spy.projections.clear()
        self.spy.enabled = True
        return self.spy

    def __exit__(self, *exc):
        self.spy.enabled = False
        return False


SPY = CommandSpy()
monitoring.register(SPY)


@contextmanager
def counted():
    """Yield the process-wide :class:`CommandSpy`, recording only in this block."""
    with SPY.capture() as spy:
        yield spy


class RepositoryTestCase(unittest.TestCase):
    """Base class that verifies the live database is actually reachable."""

    @classmethod
    def setUpClass(cls):
        get_app()
        from models import Visitor

        try:
            cls.total_visitors = Visitor.objects.count()
        except Exception as exc:  # pragma: no cover - environment problem
            raise unittest.SkipTest(f"MongoDB is not reachable: {exc}")
        if cls.total_visitors == 0:
            raise unittest.SkipTest("Visitor collection is empty")

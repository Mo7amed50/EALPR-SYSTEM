"""
ealpr/__init__.py — Application factory.

Usage:
    from ealpr import create_app
    app = create_app()
"""
import logging
import re
import time
import os
from datetime import datetime

from flask import Flask
from mongoengine import connect
from dotenv import load_dotenv

from config import (
    SECRET_KEY, MONGODB_URI, MONGODB_DB_NAME, TEMP_DIR, DEFAULT_ADMIN_PASSWORD,
    # RuntimeConfigError is re-exported so callers can catch the startup
    # rejection without importing from the root-level config module.
    RuntimeConfigError, assert_single_process, assert_production_secrets,
    resolve_runtime_mode,
)
from ealpr.extensions import login_manager, socketio

log = logging.getLogger(__name__)

#: Matches the ``user:password@`` portion of anything shaped like a connection
#: string. Startup messages interpolate driver exceptions, and a driver is free
#: to include the URI it was handed - so redaction happens on the way out, in
#: one place, rather than relying on each call site to remember.
_URI_USERINFO = re.compile(r'(?<=://)[^/@\s]+@')


def _redact_credentials(text):
    """Strip ``user:password@`` from anything about to be logged.

    Applied to every startup message that carries an operator-supplied string or
    an exception. Host names and paths stay: they are operationally useful. Only
    the credentials are removed.
    """
    return _URI_USERINFO.sub('***@', str(text))


def _ensure_startup_logging():
    """Give the application's own logger one handler, once.

    Nothing in this project configured logging before Block C5, so every INFO
    line - including the worker lifecycle messages C2/C3/C4 already emit - was
    silently discarded unless a harness was watching. Attaching a single
    handler to the ``ealpr`` logger makes them observable, and because the
    handler is scoped to that one logger, third-party loggers keep their own
    levels: this is not a global logging redesign.
    """
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(levelname)s: %(name)s: %(message)s"))
    log.addHandler(handler)
    # WARNING is the floor that matters. `basicConfig` was deliberately not
    # used: it would reconfigure the root logger for every dependency too.
    log.setLevel(logging.INFO)


def create_app() -> Flask:
    """Create, configure, and return the Flask application."""
    load_dotenv()

    # ── Block C5 (S6): make the startup path observable ─────────────────────
    # Installed first so every check below can report through the logger rather
    # than stdout, including the rejections themselves.
    _ensure_startup_logging()

    # ── Block C5 (S1): resolve the runtime mode before anything else ─────────
    # This runs before the Flask object exists so that a rejected configuration
    # fails with a clear message instead of leaving a half-built app behind. It
    # raises rather than falling back to a default: a request that cannot be
    # honoured must stop startup, not be silently downgraded.
    runtime_env, debug = resolve_runtime_mode()

    # ── Block C5 (S3): refuse a multi-process configuration ─────────────────
    # The inference queue, the one-consumer guarantee, orphaned-job recovery and
    # the live Socket.IO broadcasts are all process-local, so a second process is
    # incorrect rather than merely unsupported. Checked here because this is the
    # one place every entry path (app.py, a WSGI runner, a test) passes through.
    assert_single_process()

    # ── Block C5 (S5): refuse production with placeholder credentials ───────
    # Placed before the Flask object exists, like the checks above, and it
    # validates the very values being configured below - not a separately read
    # copy of them. Development and test are unaffected.
    assert_production_secrets(
        runtime_env,
        secret_key=SECRET_KEY,
        admin_password=DEFAULT_ADMIN_PASSWORD,
    )
    log.info("Starting EALPR in %s mode", runtime_env)

    app = Flask(
        __name__,
        # Resolve templates relative to the project root (one level up from ealpr/)
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
    )
    app.config["SECRET_KEY"] = SECRET_KEY

    # C5.1/S1: `debug` is now the validated environment switch, not a literal in
    # the entry point. It is false everywhere except an explicitly development
    # run that asked for it, which is what removes the accidental-debugger risk.
    app.debug = debug
    # Recorded so a running process can report which mode it booted in, and so
    # `test` is distinguishable from `development` without re-reading os.environ.
    app.config["EALPR_ENV"] = runtime_env
    # Read by app.py so the entry point passes the validated value to the server
    # instead of deciding debug itself.
    app.config["EALPR_DEBUG"] = debug
    if runtime_env == "test":
        app.config["TESTING"] = True

    # ── Create temp directory ────────────────────────────────────────────────
    os.makedirs(TEMP_DIR, exist_ok=True)

    # ── Initialize extensions ────────────────────────────────────────────────
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    socketio.init_app(app, async_mode="threading", cors_allowed_origins="*")

    # ── Connect to MongoDB ───────────────────────────────────────────────────
    _connect_mongodb()

    # ── User loader ──────────────────────────────────────────────────────────
    from models import User

    @login_manager.user_loader
    def load_user(user_id):
        from ealpr.extensions import DB_ENABLED
        if not DB_ENABLED:
            from bson import ObjectId
            # Return an active user when DB is offline.
            # Support admin privileges if user logged in as admin.
            is_admin = (user_id == "000000000000000000000001")
            mock_user = User(
                id=ObjectId(user_id) if ObjectId.is_valid(user_id) else ObjectId("000000000000000000000000"),
                username="admin" if is_admin else "Guest (Offline Mode)",
                is_admin=is_admin,
                is_active=True,
            )
            return mock_user
        try:
            return getattr(User, "objects")(id=user_id).max_time_ms(5000).first()
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    # ── Jinja2 filters ───────────────────────────────────────────────────────
    import base64
    from ealpr.utils import utc_to_cairo

    @app.template_filter("to_cairo")
    def to_cairo_filter(value):
        return utc_to_cairo(value)

    # ── Context Processors ───────────────────────────────────────────────────
    @app.context_processor
    def inject_db_status():
        from ealpr.extensions import DB_ENABLED
        return dict(db_enabled=DB_ENABLED)

    @app.template_filter("b64encode")
    def b64encode_filter(data):
        if data is None:
            return ""
        return base64.b64encode(data).decode("utf-8")

    @app.template_filter("detection_thumb_url")
    def detection_thumb_url_filter(detection):
        """URL of the cached preview thumbnail used by gallery cards.

        Phase 2 / Block B: gallery pages request this small derived image rather
        than the stored image, so browsing never transfers a full-resolution
        original. The original remains available through /api/images/<id>/<type>.
        """
        det_id = getattr(detection, "id", None)
        if det_id:
            return f"/api/images/{det_id}/processed/thumb"
        return ""

    @app.template_filter("detection_thumb")
    def detection_thumb_filter(detection):
        det_id = getattr(detection, "id", None)
        if det_id:
            return f"/api/images/{det_id}/processed"
        from ealpr.image_storage import encode_detection_image
        return encode_detection_image(
            getattr(detection, "processed_image_path", None),
            getattr(detection, "processed_image", None),
        )


    # ── Register blueprints ──────────────────────────────────────────────────
    from ealpr.blueprints.auth import auth_bp
    from ealpr.blueprints.main import main_bp
    from ealpr.blueprints.detection import detection_bp
    from ealpr.blueprints.visitors import visitors_bp
    from ealpr.blueprints.users import users_bp
    from ealpr.blueprints.settings import settings_bp
    from ealpr.blueprints.reports import reports_bp
    from ealpr.blueprints.inference import inference_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(visitors_bp)
    app.register_blueprint(users_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(inference_bp)

    # ── Bootstrap default admin ──────────────────────────────────────────────
    _create_default_admin()

    # ── Block C3: recover async inference jobs from a previous process ───────
    # Jobs are durable in Mongo but the work queue is not, so a restart can
    # leave jobs queued or processing forever. Recovery re-queues what is still
    # trustworthy and fails what is not. It is idempotent (one consumer, ever)
    # and is wrapped so that a recovery problem can never stop the app booting -
    # C3.10 requires startup to survive Mongo being unavailable.
    #
    # C5.4 registers the interpreter-exit hook alongside it. The hook itself is
    # idempotent, so a second create_app() in the same process (a test, a second
    # factory call) cannot stack registrations.
    try:
        from ealpr.inference_jobs import initialize_inference_jobs, register_atexit_shutdown
        register_atexit_shutdown()
        initialize_inference_jobs(app)
    except Exception as recovery_err:  # pylint: disable=broad-exception-caught
        # WARNING, not ERROR: recovery is best-effort by design and C3.10
        # requires startup to survive it. The exception text is redacted because
        # a driver failure can carry the connection string it was given.
        log.warning("inference job recovery could not start: %s",
                    _redact_credentials(recovery_err))

    return app


# ── Private helpers ──────────────────────────────────────────────────────────

def _connect_mongodb(max_retries: int = 3) -> bool:
    """Connect to MongoDB with retry and transparent offline fallback."""
    from ealpr import extensions
    from mongoengine import disconnect, get_connection

    uri = (MONGODB_URI or "").strip()
    if not uri:
        # Degraded, not fatal: C3.10 requires the app to boot read-only.
        log.warning("No MongoDB URI configured. Continuing without DB "
                    "(read-only mode).")
        extensions.DB_ENABLED = False
        return False

    if not (uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")):
        # The URI itself is deliberately not logged. At this point it has failed
        # a scheme check, so it may be a mistyped credential string rather than
        # a URL, and echoing it would put a password in the startup log.
        log.warning("MongoDB URI is missing a mongodb:// or mongodb+srv:// "
                    "scheme. Continuing without DB (read-only mode).")
        extensions.DB_ENABLED = False
        return False

    extensions.DB_ENABLED = False
    for attempt in range(max_retries):
        try:
            log.info("Connecting to MongoDB (attempt %d/%d)...",
                     attempt + 1, max_retries)
            disconnect(alias="default")
            connect(
                MONGODB_DB_NAME,
                host=uri,
                alias="default",
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                waitQueueTimeoutMS=5000,
                retryReads=False,
                retryWrites=False,
            )
            conn = get_connection(alias="default")
            conn.admin.command("ping")
            log.info("MongoDB connected to database %s.", MONGODB_DB_NAME)
            extensions.DB_ENABLED = True
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            try:
                disconnect(alias="default")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            # Driver failures frequently quote the URI they were handed, so the
            # text is redacted before it reaches the log.
            log.warning("MongoDB attempt %d failed: %s",
                        attempt + 1, _redact_credentials(e))
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

            log.error("All MongoDB connection attempts failed. Continuing "
                      "without DB (read-only mode).")
            extensions.DB_ENABLED = False
            return False


def _create_default_admin() -> None:
    """Ensure a default admin account exists (username: admin, configurable password)."""
    from ealpr.extensions import DB_ENABLED

    if not DB_ENABLED:
        return

    try:
        from models import User
        admin = getattr(User, "objects")(username="admin").max_time_ms(5000).first()
        if not admin:
            admin = User(
                username="admin",
                is_admin=True,
                created_at=datetime.utcnow(),
            )
            admin.set_password(DEFAULT_ADMIN_PASSWORD)
            admin.save()
            # Status only. The password itself is never logged, at any level.
            log.info("Created default admin user.")
            if DEFAULT_ADMIN_PASSWORD == "admin123":
                log.warning("DEFAULT_ADMIN_PASSWORD is still the development "
                            "default. Change it before production - S5 refuses "
                            "to start in production while it is unchanged.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # WARNING: a failed bootstrap leaves the existing account untouched and
        # does not stop the app booting.
        log.warning("could not create default admin: %s",
                    _redact_credentials(e))

"""
ealpr/__init__.py — Application factory.

Usage:
    from ealpr import create_app
    app = create_app()
"""
import time
import os
from datetime import datetime

from flask import Flask
from mongoengine import connect
from dotenv import load_dotenv

from config import (
    SECRET_KEY, MONGODB_URI, MONGODB_DB_NAME, TEMP_DIR, DEFAULT_ADMIN_PASSWORD,
)
from ealpr.extensions import login_manager, socketio


def create_app() -> Flask:
    """Create, configure, and return the Flask application."""
    load_dotenv()

    app = Flask(
        __name__,
        # Resolve templates relative to the project root (one level up from ealpr/)
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
    )
    app.config["SECRET_KEY"] = SECRET_KEY

    # ── Create temp directory ────────────────────────────────────────────────
    os.makedirs(TEMP_DIR, exist_ok=True)

    # ── Initialize extensions ────────────────────────────────────────────────
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    socketio.init_app(app)

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

    @app.template_filter("detection_thumb")
    def detection_thumb_filter(detection):
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

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(visitors_bp)
    app.register_blueprint(users_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(reports_bp)

    # ── Bootstrap default admin ──────────────────────────────────────────────
    _create_default_admin()

    return app


# ── Private helpers ──────────────────────────────────────────────────────────

def _connect_mongodb(max_retries: int = 3) -> bool:
    """Connect to MongoDB with exponential back-off retry."""
    from ealpr import extensions
    from mongoengine import disconnect, get_connection

    if not MONGODB_URI:
        print("No MongoDB URI configured. Continuing without DB (read-only mode).")
        extensions.DB_ENABLED = False
        return False

    for attempt in range(max_retries):
        try:
            print(f"Connecting to MongoDB (attempt {attempt + 1}/{max_retries})...")
            connect(
                MONGODB_DB_NAME,
                host=MONGODB_URI,
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
            print("MongoDB connected!")
            extensions.DB_ENABLED = True
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"MongoDB attempt {attempt + 1} failed: {e}")
            try:
                disconnect(alias="default")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print("All connection attempts failed. Continuing without DB (read-only mode).")
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
            print("Created default admin user.")
            if DEFAULT_ADMIN_PASSWORD == "admin123":
                print("Warning: change DEFAULT_ADMIN_PASSWORD in production.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: could not create default admin: {e}")

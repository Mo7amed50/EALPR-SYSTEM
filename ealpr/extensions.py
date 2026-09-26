"""
extensions.py — Shared Flask extension instances.

These objects are instantiated here (unbound) and initialized
inside create_app() via the init_app() pattern to avoid circular imports.
"""
from flask_login import LoginManager
from flask_socketio import SocketIO

# Unbound extension instances — initialized in create_app()
login_manager = LoginManager()

# async_mode='threading' is required when running under the standard Flask/Werkzeug
# dev server (no eventlet/gevent installed).  Without it, Flask-SocketIO falls back
# to eventlet which conflicts with the regular threading model and causes the whole
# app to freeze when any background thread tries to do I/O.
socketio = SocketIO(async_mode="threading", cors_allowed_origins="*")

# Global database state flag
DB_ENABLED = False

"""
extensions.py — Shared Flask extension instances.

These objects are instantiated here (unbound) and initialized
inside create_app() via the init_app() pattern to avoid circular imports.
"""
from flask_login import LoginManager
from flask_socketio import SocketIO

# Unbound extension instances — initialized in create_app()
login_manager = LoginManager()
socketio = SocketIO()

# Global database state flag
DB_ENABLED = False

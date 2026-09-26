"""
app.py — Entry point for the EALPR system.

The application is assembled by the factory in ealpr/__init__.py.
Run with:  python app.py
"""
from ealpr import create_app
from ealpr.extensions import socketio

app = create_app()

if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
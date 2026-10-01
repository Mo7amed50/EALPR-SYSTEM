"""
app.py — Entry point for the EALPR system.

The application is assembled by the factory in ealpr/__init__.py.
Run with:  python app.py

Phase 2 / Block C5 (S1, S3, S4):
  * `debug` is no longer hard-coded. The factory resolves it from EALPR_ENV /
    EALPR_DEBUG and refuses an invalid combination, so this entry point only
    reads the value the factory already validated.
  * `use_reloader=False` stays explicit. It is not cosmetic: the reloader starts
    a second process, and this app's inference queue, one-consumer guarantee and
    live Socket.IO broadcasts are all process-local. See
    `config.assert_single_process`.
  * `allow_unsafe_werkzeug=True` is the documented Werkzeug behaviour for a
    development run and is unchanged. Selecting a production WSGI server is S2 and
    is deliberately out of scope for this block.
"""
from ealpr import create_app
from ealpr.extensions import socketio

app = create_app()

if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        # Validated in create_app(): True only for an explicit development run.
        debug=app.config.get("EALPR_DEBUG", app.debug),
        # Never True. A second process is incorrect here, not merely untested.
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ealpr import create_app
from ealpr.extensions import socketio

app = create_app()
port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
socketio.run(app, host="127.0.0.1", port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
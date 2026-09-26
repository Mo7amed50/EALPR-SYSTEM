import sys, os, time
sys.path.insert(0, r"D:\EALPR SYSTEM")
os.chdir(r"D:\EALPR SYSTEM")

print("FPS_LIMIT env:", os.environ.get("FPS_LIMIT"))
from config import FPS_LIMIT, FRAME_WIDTH, FRAME_HEIGHT
print("config FPS_LIMIT:", FPS_LIMIT, "FRAME:", FRAME_WIDTH, FRAME_HEIGHT)

import cv2
import ealpr.utils as u

# Instrument the worker's encode rate indirectly by counting notifications.
feed_holder = {"count": 0}

orig_time = time.time
t0 = time.time()

import threading
out = []

refs, feed = u._get_or_start_feed()
print("refs:", refs, "feed alive:", feed["alive"] if feed else None)
print("fps_limit:", feed["fps_limit"], "frame_skip:", feed["frame_skip"])

def reader():
    last = 0
    while time.time() - t0 < 5:
        with feed["condition"]:
            if feed["frame"] is not None and feed["frame"] is not last:
                out.append(1)
                last = feed["frame"]
        time.sleep(0.01)

r = threading.Thread(target=reader, daemon=True)
r.start()
time.sleep(5)
print("frames observed in 5s:", len(out))
u._release_feed(feed)
time.sleep(0.5)
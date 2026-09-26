import sys, os, time, threading
sys.path.insert(0, r"D:\EALPR SYSTEM")
os.chdir(r"D:\EALPR SYSTEM")

import cv2

enc_count = [0]

orig_imencode = cv2.imencode
def spy_imencode(*a, **k):
    enc_count[0] += 1
    return orig_imencode(*a, **k)
cv2.imencode = spy_imencode

import ealpr.utils as u
from config import FPS_LIMIT

u.get_video_frame_skip = lambda: 999999  # disable detection to isolate encode path

t0 = time.time()
refs, feed = u._get_or_start_feed()
print("fps_limit:", feed["fps_limit"])

time.sleep(5)
print("imencode calls in 5s:", enc_count[0])
u._release_feed(feed)
time.sleep(0.5)
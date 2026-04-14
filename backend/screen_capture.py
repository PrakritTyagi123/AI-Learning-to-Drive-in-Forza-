"""
ForzaTek AI — Screen Capture
Grabs the game screen using mss, encodes with OpenCV, returns base64 JPEG.
Also provides raw numpy frames for AI inference.
"""

import base64
import time
import threading

try:
    import mss
    import cv2
    import numpy as np
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

from config import CAPTURE_FPS, CAPTURE_QUALITY, CAPTURE_MONITOR, CAPTURE_RESIZE


class ScreenCapture:
    def __init__(self):
        self.running = False
        self.frame_b64 = None
        self.frame_raw = None      # NEW: raw numpy BGR frame for AI inference
        self.lock = threading.Lock()
        self.thread = None
        self.fps_actual = 0

    @property
    def available(self):
        return MSS_AVAILABLE

    def start(self):
        if not MSS_AVAILABLE:
            print("[CAPTURE] mss/opencv not available — screen capture disabled")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[CAPTURE] Started — target {CAPTURE_FPS} FPS, quality {CAPTURE_QUALITY}%, monitor {CAPTURE_MONITOR}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_frame(self) -> str | None:
        with self.lock:
            return self.frame_b64

    def get_frame_raw(self):
        """Return raw numpy BGR frame for AI inference."""
        with self.lock:
            return self.frame_raw.copy() if self.frame_raw is not None else None

    def _capture_loop(self):
        interval = 1.0 / CAPTURE_FPS
        frame_count = 0
        fps_timer = time.time()

        with mss.mss() as sct:
            monitors = sct.monitors
            if CAPTURE_MONITOR >= len(monitors):
                monitor = monitors[1] if len(monitors) > 1 else monitors[0]
            else:
                monitor = monitors[CAPTURE_MONITOR]

            print(f"[CAPTURE] Monitor: {monitor['width']}x{monitor['height']}")

            while self.running:
                t0 = time.time()
                try:
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    if CAPTURE_RESIZE:
                        frame = cv2.resize(frame, CAPTURE_RESIZE, interpolation=cv2.INTER_AREA)

                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, CAPTURE_QUALITY])
                    b64 = base64.b64encode(buffer).decode('utf-8')

                    with self.lock:
                        self.frame_b64 = b64
                        self.frame_raw = frame  # Store raw frame for AI

                    frame_count += 1
                    elapsed = time.time() - fps_timer
                    if elapsed >= 1.0:
                        self.fps_actual = frame_count / elapsed
                        frame_count = 0
                        fps_timer = time.time()

                except Exception as e:
                    print(f"[CAPTURE] Error: {e}")
                    time.sleep(0.5)

                dt = time.time() - t0
                sleep_time = interval - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

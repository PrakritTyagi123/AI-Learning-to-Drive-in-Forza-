"""
ForzaTek AI — Live Forza Recorder
==================================
Captures frames from the live game feed with perceptual-hash deduplication
and diversity bucket tracking. Writes to the shared database.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.database import DB_PATH, init_db, write_conn, read_conn

PHASH_HAMMING_THRESHOLD = 6
PHASH_WINDOW_SIZE = 500
MIN_SAVE_INTERVAL_SEC = 0.25
JPEG_QUALITY = 85
TARGET_PER_BUCKET = 40


def compute_phash(frame_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    v = 0
    for b in diff.flatten():
        v = (v << 1) | int(b)
    return v


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def infer_time_of_day(frame_bgr: np.ndarray) -> str:
    h = frame_bgr.shape[0]
    sky = frame_bgr[: h // 3]
    luma = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY).mean()
    if luma < 40: return "night"
    if luma < 90: return "dusk"
    if luma < 160: return "day"
    return "bright_day"


def infer_weather(frame_bgr: np.ndarray) -> str:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean()
    val = hsv[:, :, 2].mean()
    if sat < 35 and val > 180: return "fog_or_snow"
    if sat < 55: return "overcast"
    return "clear"


@dataclass
class SessionStats:
    session_start: float = field(default_factory=time.time)
    frames_received: int = 0
    frames_saved: int = 0
    frames_skipped_dedup: int = 0
    frames_skipped_interval: int = 0
    disk_bytes: int = 0
    bucket_fill: dict = field(default_factory=dict)
    recording: bool = False
    game_version: str = "fh4"
    biome_override: Optional[str] = None

    def snapshot(self) -> dict:
        d = asdict(self)
        d["elapsed_sec"] = round(time.time() - self.session_start, 1)
        d["disk_mb"] = round(self.disk_bytes / (1024 * 1024), 1)
        d["target_per_bucket"] = TARGET_PER_BUCKET
        d["buckets_at_target"] = sum(1 for v in self.bucket_fill.values() if v >= TARGET_PER_BUCKET)
        d["total_buckets"] = max(1, len(self.bucket_fill))
        return d


class SmartRecorder:
    """Thread-safe live recorder. Single instance per app."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_db(db_path)
        self._lock = threading.Lock()
        self._recent_hashes: list[int] = []
        self._last_save = 0.0
        self.stats = SessionStats()

    @staticmethod
    def _bucket_key(gv, biome, weather, tod):
        return f"{gv}/{biome}/{weather}/{tod}"

    def _reload_buckets(self):
        with read_conn(self.db_path) as c:
            rows = c.execute("""
                SELECT game_version, biome, weather, time_of_day, COUNT(*)
                FROM frames WHERE source_type='live' GROUP BY 1,2,3,4
            """).fetchall()
            self.stats.bucket_fill = {
                self._bucket_key(r[0], r[1], r[2], r[3]): r[4] for r in rows
            }

    def start(self, game_version="fh4", biome_override=None):
        with self._lock:
            self.stats = SessionStats(
                game_version=game_version,
                biome_override=biome_override,
                recording=True,
            )
            self._reload_buckets()
            self._recent_hashes.clear()
            self._last_save = 0.0

    def stop(self):
        with self._lock:
            self.stats.recording = False

    def set_biome(self, biome: str):
        with self._lock:
            self.stats.biome_override = biome

    def maybe_save(self, frame_bgr: np.ndarray, telemetry: dict) -> bool:
        with self._lock:
            if not self.stats.recording:
                return False
            self.stats.frames_received += 1

            now = time.time()
            if now - self._last_save < MIN_SAVE_INTERVAL_SEC:
                self.stats.frames_skipped_interval += 1
                return False

            ph = compute_phash(frame_bgr) & 0x7FFFFFFFFFFFFFFF
            for h in self._recent_hashes:
                if hamming_distance(ph, h) <= PHASH_HAMMING_THRESHOLD:
                    self.stats.frames_skipped_dedup += 1
                    return False

            biome = self.stats.biome_override or "unknown"
            weather = infer_weather(frame_bgr)
            tod = infer_time_of_day(frame_bgr)
            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                return False
            jpeg = buf.tobytes()
            h_px, w_px = frame_bgr.shape[:2]

            with write_conn(self.db_path) as c:
                c.execute(
                    """INSERT INTO frames
                       (ts, source_type, game_version, biome, weather, time_of_day,
                        phash, frame_jpeg, width, height, telemetry_json)
                       VALUES (?, 'live', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (now, self.stats.game_version, biome, weather, tod, ph,
                     jpeg, w_px, h_px, json.dumps(telemetry))
                )

            self._recent_hashes.append(ph)
            if len(self._recent_hashes) > PHASH_WINDOW_SIZE:
                self._recent_hashes.pop(0)
            self._last_save = now
            self.stats.frames_saved += 1
            self.stats.disk_bytes += len(jpeg)
            key = self._bucket_key(self.stats.game_version, biome, weather, tod)
            self.stats.bucket_fill[key] = self.stats.bucket_fill.get(key, 0) + 1
            return True

    def get_stats(self) -> dict:
        with self._lock:
            return self.stats.snapshot()

    def get_bucket_report(self) -> list[dict]:
        with self._lock:
            out = []
            for k, count in self.stats.bucket_fill.items():
                gv, biome, weather, tod = k.split("/")
                out.append({
                    "game_version": gv, "biome": biome, "weather": weather,
                    "time_of_day": tod, "count": count,
                    "target": TARGET_PER_BUCKET,
                    "deficit": max(0, TARGET_PER_BUCKET - count),
                })
            out.sort(key=lambda r: (-r["deficit"], r["count"]))
            return out


smart_recorder = SmartRecorder()


def register_routes(app):
    from fastapi import Body

    @app.post("/api/record/start")
    def _start(payload: dict = Body(...)):
        smart_recorder.start(
            game_version=payload.get("game_version", "fh4"),
            biome_override=payload.get("biome_override"),
        )
        return {"ok": True, "stats": smart_recorder.get_stats()}

    @app.post("/api/record/stop")
    def _stop():
        smart_recorder.stop()
        return {"ok": True, "stats": smart_recorder.get_stats()}

    @app.post("/api/record/biome")
    def _biome(payload: dict = Body(...)):
        smart_recorder.set_biome(payload["biome"])
        return {"ok": True}

    @app.get("/api/record/stats")
    def _stats():
        return smart_recorder.get_stats()

    @app.get("/api/record/buckets")
    def _buckets():
        return {"buckets": smart_recorder.get_bucket_report()}

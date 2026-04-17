"""
ForzaTek AI — Telemetry Listener
=================================
UDP socket listener that parses Forza Motorsport / Horizon "Data Out · dash"
packets and exposes the latest snapshot + rolling 60 s history over HTTP.

Forza ships a binary struct with 323 bytes in the "dash" layout. We decode
the fields we actually need for recording + the telemetry dashboard.

The listener runs in a daemon thread so it never blocks the HTTP server.
If no packet has arrived yet, /api/telemetry/live returns {"speed": null, ...}
and the UI shows "NO SIGNAL".
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI

# ─── UDP + decoding ─────────────────────────────────────────────────────

# Full Forza dash format layout. We only read the fields the app uses.
# References: Microsoft's Forza Data Out documentation.
#
# Field definitions (dash format):
#   IsRaceOn            0    s32
#   TimestampMS         4    u32
#   EngineMaxRpm        8    f32
#   EngineIdleRpm      12    f32
#   CurrentEngineRpm   16    f32
#   Acceleration X/Y/Z 20    3x f32
#   Velocity X/Y/Z     32    3x f32
#   AngularVel X/Y/Z   44    3x f32
#   Yaw, Pitch, Roll   56    3x f32
#   Normalized suspension (wheels) and tire slip follow…
#   CarOrdinal        212    s32
#   CarClass          216    s32
#   CarPerformanceIdx 220    s32
#   DrivetrainType    224    s32
#   NumCylinders      228    s32
#   PositionX         232    f32
#   PositionY         236    f32
#   PositionZ         240    f32
#   Speed             244    f32   (m/s)
#   Power             248    f32
#   Torque            252    f32
#   TireTempFL        256    f32
#   TireTempFR        260    f32
#   TireTempRL        264    f32
#   TireTempRR        268    f32
#   Boost             272    f32
#   Fuel              276    f32
#   DistanceTraveled  280    f32
#   BestLap           284    f32
#   LastLap           288    f32
#   CurrentLap        292    f32
#   CurrentRaceTime   296    f32
#   LapNumber         300    u16
#   RacePosition      302    u8
#   Accel             303    u8
#   Brake             304    u8
#   Clutch            305    u8
#   HandBrake         306    u8
#   Gear              307    u8
#   Steer             308    s8
#   NormalDrivingLine 309    s8
#   NormalAIBrakeDiff 310    s8

DASH_STRUCT_PREFIX = struct.Struct("<i I f f f 3f 3f 3f 3f")  # first 56 bytes


def parse_dash_packet(data: bytes) -> Optional[Dict[str, Any]]:
    """Parse a Forza Data Out "dash" packet (324 bytes). Returns a dict of
    decoded fields, or None if the packet is too small."""
    if len(data) < 311:
        return None
    try:
        # Header / motion
        (is_race_on, ts_ms, max_rpm, idle_rpm, cur_rpm,
         ax, ay, az, vx, vy, vz, _ax, _ay, _az, yaw, pitch, roll) = \
            DASH_STRUCT_PREFIX.unpack_from(data, 0)

        # Scalar fields at known offsets
        pos_x = struct.unpack_from("<f", data, 232)[0]
        pos_y = struct.unpack_from("<f", data, 236)[0]
        pos_z = struct.unpack_from("<f", data, 240)[0]
        speed_mps = struct.unpack_from("<f", data, 244)[0]

        tire_fl = struct.unpack_from("<f", data, 256)[0]
        tire_fr = struct.unpack_from("<f", data, 260)[0]
        tire_rl = struct.unpack_from("<f", data, 264)[0]
        tire_rr = struct.unpack_from("<f", data, 268)[0]
        fuel = struct.unpack_from("<f", data, 276)[0]
        best_lap = struct.unpack_from("<f", data, 284)[0]
        last_lap = struct.unpack_from("<f", data, 288)[0]

        lap_number = struct.unpack_from("<H", data, 300)[0]
        race_position = struct.unpack_from("<B", data, 302)[0]
        accel_u8 = struct.unpack_from("<B", data, 303)[0]
        brake_u8 = struct.unpack_from("<B", data, 304)[0]
        gear_u8  = struct.unpack_from("<B", data, 307)[0]
        steer_s8 = struct.unpack_from("<b", data, 308)[0]

        drivetrain_id = struct.unpack_from("<i", data, 224)[0]
        drivetrain_map = {0: "FWD", 1: "RWD", 2: "AWD"}

        return {
            "ts": time.time(),
            "is_race_on": bool(is_race_on),
            "speed": speed_mps * 2.23694,  # to MPH
            "speed_mps": speed_mps,
            "rpm": cur_rpm,
            "redline": max_rpm,
            "gear": gear_u8,
            "throttle": accel_u8 / 255.0,
            "brake": brake_u8 / 255.0,
            "steer": steer_s8 / 127.0,
            "pos_x": pos_x,
            "pos_y": pos_z,      # Forza's Z axis is the ground plane
            "pos_z": pos_y,
            "heading": (yaw * 180.0 / 3.14159265) % 360.0,
            "g_lat":  ax / 9.80665,
            "g_long": az / 9.80665,
            "tire_temp": (tire_fl + tire_fr + tire_rl + tire_rr) / 4.0,
            "fuel": max(0.0, min(1.0, fuel)),
            "lap": lap_number,
            "lap_total": None,   # not present in dash format
            "last_lap": last_lap if last_lap > 0 else None,
            "best_lap": best_lap if best_lap > 0 else None,
            "drivetrain": drivetrain_map.get(drivetrain_id, "—"),
            "car": None,  # not in dash packet
        }
    except (struct.error, IndexError):
        return None


# ─── Buffer ───────────────────────────────────────────────────────────────

class TelemetryBuffer:
    """Thread-safe rolling buffer of the last ~60 s of telemetry samples."""

    def __init__(self, max_seconds: float = 60.0, max_samples: int = 4000):
        self._lock = threading.Lock()
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=max_samples)
        self.max_seconds = max_seconds
        self.port: int = 5300
        self._last_packet_ts: float = 0.0
        self._rate_count = 0
        self._rate_start = time.time()
        self._rate_hz = 0.0

    def push(self, sample: Dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(sample)
            cutoff = sample["ts"] - self.max_seconds
            while self._buf and self._buf[0]["ts"] < cutoff:
                self._buf.popleft()
            self._last_packet_ts = sample["ts"]
            self._rate_count += 1
            elapsed = sample["ts"] - self._rate_start
            if elapsed >= 1.0:
                self._rate_hz = self._rate_count / elapsed
                self._rate_count = 0
                self._rate_start = sample["ts"]

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._buf:
                return None
            s = dict(self._buf[-1])
        s["port"] = self.port
        s["rate_hz"] = self._rate_hz
        return s

    def history(self) -> list:
        with self._lock:
            return list(self._buf)

    def stale_sec(self) -> float:
        if not self._last_packet_ts:
            return float("inf")
        return time.time() - self._last_packet_ts


BUFFER = TelemetryBuffer()


# ─── UDP listener thread ──────────────────────────────────────────────────

_listener_started = False


def _listen_loop(port: int) -> None:
    """Blocking receive loop — run in a daemon thread."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        # Port busy — give up silently; the dashboard will just show NO SIGNAL.
        return
    sock.settimeout(1.0)
    BUFFER.port = port

    while True:
        try:
            data, _addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break
        sample = parse_dash_packet(data)
        if sample is None:
            continue
        BUFFER.push(sample)


def start_background_listener(port: int = 5300) -> None:
    global _listener_started
    if _listener_started:
        return
    _listener_started = True
    t = threading.Thread(target=_listen_loop, args=(port,), daemon=True, name="forza-udp")
    t.start()


# ─── Routes ───────────────────────────────────────────────────────────────

def register_routes(app: FastAPI) -> None:
    @app.get("/api/telemetry/live")
    def live():
        s = BUFFER.latest()
        if s is None:
            return {"speed": None}
        # If packets haven't arrived in >3 s, treat as stale
        if BUFFER.stale_sec() > 3.0:
            return {"speed": None, "stale": True}
        return s

    @app.get("/api/telemetry/history")
    def history():
        return {"samples": BUFFER.history(), "rate_hz": BUFFER._rate_hz}

    @app.post("/api/telemetry/push_fake")
    async def push_fake(payload: dict):
        """Test-only endpoint: push a synthetic telemetry sample into the buffer."""
        payload = dict(payload or {})
        payload.setdefault("ts", time.time())
        BUFFER.push(payload)
        return {"ok": True}

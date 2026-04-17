"""
Tests for backend/telemetry_listener.py.

Run with: python -m tests.test_telemetry

Covers:
  • Forza dash packet parsing (synthesized via the same struct layout)
  • Buffer rolls by time, trims to ~60 s
  • Rate_hz tracking
  • Latest snapshot is read back correctly
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.telemetry_listener import (
    parse_dash_packet, TelemetryBuffer, BUFFER,
)


# ─── Synthetic packet builder ────────────────────────────────────────────

def _build_dash_packet(
    speed_mps: float = 60.0,
    rpm: float = 6500.0,
    max_rpm: float = 7200.0,
    throttle_u8: int = 220,
    brake_u8: int = 10,
    gear_u8: int = 5,
    steer_s8: int = -30,
    pos_x: float = 100.0,
    pos_z: float = 200.0,   # Forza's Z is the ground-plane axis we map to "y"
    pos_y: float = 5.0,
    yaw: float = 1.2,
    ax: float = 0.3,
    az: float = -0.5,
    tire_temp: float = 85.0,
    fuel: float = 0.75,
    best_lap: float = 98.3,
    last_lap: float = 101.1,
    lap_number: int = 2,
    drivetrain: int = 1,   # RWD
) -> bytes:
    """Build a 324-byte packet matching the layout parse_dash_packet expects."""
    buf = bytearray(324)
    # Prefix struct: < i I f f f 3f 3f 3f 3f
    struct.pack_into(
        "<i I f f f 3f 3f 3f 3f", buf, 0,
        1,                    # IsRaceOn
        int(time.time() * 1000) & 0xFFFFFFFF,
        max_rpm, 800.0, rpm,  # max / idle / current rpm
        ax, 0.0, az,          # accel x/y/z
        0.0, 0.0, speed_mps,  # velocity x/y/z
        0.0, 0.0, 0.0,        # angular vel
        yaw, 0.0, 0.0,        # yaw/pitch/roll
    )
    struct.pack_into("<i",  buf, 224, drivetrain)
    struct.pack_into("<f",  buf, 232, pos_x)
    struct.pack_into("<f",  buf, 236, pos_y)
    struct.pack_into("<f",  buf, 240, pos_z)
    struct.pack_into("<f",  buf, 244, speed_mps)
    struct.pack_into("<f",  buf, 256, tire_temp)
    struct.pack_into("<f",  buf, 260, tire_temp)
    struct.pack_into("<f",  buf, 264, tire_temp)
    struct.pack_into("<f",  buf, 268, tire_temp)
    struct.pack_into("<f",  buf, 276, fuel)
    struct.pack_into("<f",  buf, 284, best_lap)
    struct.pack_into("<f",  buf, 288, last_lap)
    struct.pack_into("<H",  buf, 300, lap_number)
    struct.pack_into("<B",  buf, 302, 3)        # race position
    struct.pack_into("<B",  buf, 303, throttle_u8)
    struct.pack_into("<B",  buf, 304, brake_u8)
    struct.pack_into("<B",  buf, 307, gear_u8)
    struct.pack_into("<b",  buf, 308, steer_s8)
    return bytes(buf)


# ─── Packet parsing ──────────────────────────────────────────────────────

def test_parse_too_short_returns_none():
    assert parse_dash_packet(b"short") is None
    print("✓ parse_dash_packet rejects too-short input")


def test_parse_valid_packet():
    pkt = _build_dash_packet(
        speed_mps=60.0, rpm=6500.0, throttle_u8=220, brake_u8=10,
        gear_u8=5, steer_s8=-30, drivetrain=1,
    )
    s = parse_dash_packet(pkt)
    assert s is not None, "Valid packet should parse"
    # Speed converted to MPH
    assert abs(s["speed"] - (60.0 * 2.23694)) < 0.5, f"speed decode wrong: {s['speed']}"
    assert abs(s["rpm"] - 6500.0) < 0.1
    assert s["gear"] == 5
    assert abs(s["throttle"] - 220/255.0) < 1e-4
    assert abs(s["brake"] - 10/255.0) < 1e-4
    assert abs(s["steer"] - (-30/127.0)) < 1e-3
    assert s["drivetrain"] == "RWD"
    assert s["is_race_on"] is True
    assert s["lap"] == 2
    print("✓ parse_dash_packet decodes speed / rpm / gear / inputs correctly")


def test_parse_inactive_lap_times_become_none():
    pkt = _build_dash_packet(best_lap=0.0, last_lap=0.0)
    s = parse_dash_packet(pkt)
    assert s["best_lap"] is None
    assert s["last_lap"] is None
    print("✓ lap times of 0 decode to None (Forza sends 0 when no lap logged)")


# ─── Rolling buffer ──────────────────────────────────────────────────────

def test_buffer_latest_is_most_recent():
    b = TelemetryBuffer(max_seconds=60, max_samples=1000)
    now = time.time()
    b.push({"ts": now - 2.0, "speed": 30.0})
    b.push({"ts": now - 1.0, "speed": 60.0})
    b.push({"ts": now - 0.1, "speed": 90.0})
    latest = b.latest()
    assert latest is not None
    assert latest["speed"] == 90.0, f"expected 90, got {latest['speed']}"
    assert "rate_hz" in latest and "port" in latest
    print("✓ TelemetryBuffer.latest() returns most recent sample with metadata")


def test_buffer_drops_stale_samples():
    b = TelemetryBuffer(max_seconds=10, max_samples=1000)
    now = time.time()
    # Insert one ancient sample then a fresh one
    b.push({"ts": now - 120.0, "speed": 1.0})
    b.push({"ts": now,          "speed": 2.0})
    hist = b.history()
    assert len(hist) == 1, f"old sample should have been trimmed, got {len(hist)} entries"
    assert hist[0]["speed"] == 2.0
    print("✓ TelemetryBuffer drops samples older than max_seconds")


def test_buffer_enforces_max_samples():
    b = TelemetryBuffer(max_seconds=3600, max_samples=10)
    for i in range(50):
        b.push({"ts": time.time() + i * 0.001, "speed": float(i)})
    assert len(b.history()) == 10
    print("✓ TelemetryBuffer respects max_samples cap")


def test_stale_detection():
    b = TelemetryBuffer()
    assert b.stale_sec() == float("inf"), "empty buffer should report inf staleness"
    b.push({"ts": time.time(), "speed": 10.0})
    assert b.stale_sec() < 1.0, "fresh buffer should report tiny staleness"
    print("✓ stale_sec() behaves correctly")


def test_shared_buffer_instance():
    """The module-level BUFFER is the one the FastAPI routes read from."""
    assert isinstance(BUFFER, TelemetryBuffer)
    print("✓ module-level BUFFER is a TelemetryBuffer")


if __name__ == "__main__":
    test_parse_too_short_returns_none()
    test_parse_valid_packet()
    test_parse_inactive_lap_times_become_none()
    test_buffer_latest_is_most_recent()
    test_buffer_drops_stale_samples()
    test_buffer_enforces_max_samples()
    test_stale_detection()
    test_shared_buffer_instance()
    print("\nAll telemetry tests passed.")

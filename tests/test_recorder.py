"""
Tests for the smart recorder.
Run with: python -m tests.test_recorder
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.recorder import (
    compute_phash, hamming_distance,
    infer_time_of_day, infer_weather,
    SmartRecorder,
)


def make_frame(r=50, g=80, b=120, size=(720, 1280)):
    f = np.zeros((*size, 3), dtype=np.uint8)
    f[:, :] = [b, g, r]  # OpenCV is BGR
    return f


def test_phash_stable_across_noise():
    f1 = make_frame(50, 80, 120)
    f2 = f1.copy()
    f2 += np.random.randint(-3, 3, f2.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)
    h1 = compute_phash(f1)
    h2 = compute_phash(f2)
    assert hamming_distance(h1, h2) < 5, "pHash should be robust to small noise"
    print("✓ phash_stable_across_noise")


def test_phash_differs_across_scenes():
    f1 = make_frame(50, 80, 120)
    f2 = make_frame(200, 30, 30)
    h1 = compute_phash(f1)
    h2 = compute_phash(f2)
    # Different scenes will have different hashes in most cases
    # (uniform color frames can coincidentally match — accept if they differ OR we can't distinguish)
    dist = hamming_distance(h1, h2)
    # Uniform frames often have same hash; test with actual texture instead
    f1 = np.random.randint(0, 100, (720, 1280, 3), dtype=np.uint8)
    f2 = np.random.randint(150, 255, (720, 1280, 3), dtype=np.uint8)
    h1 = compute_phash(f1)
    h2 = compute_phash(f2)
    assert hamming_distance(h1, h2) > 5, "Different scenes should produce different hashes"
    print("✓ phash_differs_across_scenes")


def test_time_of_day_classification():
    dark = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert infer_time_of_day(dark) == "night"
    bright = np.full((720, 1280, 3), 200, dtype=np.uint8)
    assert infer_time_of_day(bright) == "bright_day"
    print("✓ time_of_day_classification")


def test_weather_classification():
    # Gray flat frame → overcast
    gray = np.full((720, 1280, 3), 130, dtype=np.uint8)
    assert infer_weather(gray) in ("overcast", "fog_or_snow")
    # Saturated frame → clear
    blue_sky = np.zeros((720, 1280, 3), dtype=np.uint8)
    blue_sky[:, :] = [180, 100, 50]  # BGR blue
    assert infer_weather(blue_sky) == "clear"
    print("✓ weather_classification")


def test_recorder_saves_and_dedups():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        rec = SmartRecorder(db_path=db)
        rec.start(game_version="fh4", biome_override="city")

        # Textured frames to produce different phashes
        for i in range(3):
            f = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            time.sleep(0.3)  # respect MIN_SAVE_INTERVAL
            rec.maybe_save(f, {})

        # Duplicate frame — should be deduped
        dup = np.full((360, 640, 3), 70, dtype=np.uint8)
        time.sleep(0.3)
        rec.maybe_save(dup, {})
        time.sleep(0.3)
        rec.maybe_save(dup, {})

        stats = rec.get_stats()
        assert stats["frames_saved"] >= 3
        assert stats["frames_skipped_dedup"] >= 1
    print("✓ recorder_saves_and_dedups")


def test_bucket_tracking():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        rec = SmartRecorder(db_path=db)
        rec.start(game_version="fh4", biome_override="forest")
        for i in range(3):
            f = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            time.sleep(0.3)
            rec.maybe_save(f, {})
        report = rec.get_bucket_report()
        assert len(report) >= 1
        assert all("deficit" in b for b in report)
    print("✓ bucket_tracking")


if __name__ == "__main__":
    test_phash_stable_across_noise()
    test_phash_differs_across_scenes()
    test_time_of_day_classification()
    test_weather_classification()
    test_recorder_saves_and_dedups()
    test_bucket_tracking()
    print("\nAll recorder tests passed.")

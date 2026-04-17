"""
Tests for video_ingester.

Run with: python -m tests.test_video_ingester

We deliberately do NOT hit YouTube or require yt-dlp in these unit tests.
We test the pure image-processing helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.video_ingester import apply_hud_mask, looks_like_non_gameplay


# ─── Fixtures ─────────────────────────────────────────────────────────────

def make_frame(r=120, g=100, b=80, size=(720, 1280)):
    """Typical-looking gameplay frame: varied content over a colored base."""
    f = np.zeros((*size, 3), dtype=np.uint8)
    f[:, :] = [b, g, r]
    # Add strong texture so the variance filter considers this gameplay.
    # The filter requires global std >= 12 and center std >= 8.
    rng = np.random.RandomState(0)
    noise = rng.randint(0, 120, f.shape, dtype=np.int16)
    f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return f


# ─── HUD mask tests ──────────────────────────────────────────────────────

def test_apply_hud_mask_zeros_the_rectangle():
    f = make_frame()
    masked = apply_hud_mask(f, [{"x": 0.05, "y": 0.80, "w": 0.20, "h": 0.15}])
    H, W = f.shape[:2]
    x1 = int(0.05 * W); y1 = int(0.80 * H)
    x2 = int(0.25 * W); y2 = int(0.95 * H)
    # The masked region must be all zeros
    assert (masked[y1:y2, x1:x2] == 0).all(), "HUD mask did not zero-out the rectangle"
    # A region well outside the mask must be unchanged
    assert np.array_equal(masked[0:10, 0:10], f[0:10, 0:10]), \
        "Pixels outside the mask were unexpectedly modified"
    print("✓ apply_hud_mask zeros the rectangle and leaves other pixels alone")


def test_apply_hud_mask_multiple_rects():
    f = make_frame()
    rects = [
        {"x": 0.00, "y": 0.00, "w": 0.10, "h": 0.10},  # top-left
        {"x": 0.80, "y": 0.80, "w": 0.15, "h": 0.15},  # bottom-right
    ]
    masked = apply_hud_mask(f, rects)
    # First rect: 0..0.1 of 720×1280 = rows 0..72, cols 0..128
    assert (masked[0:72, 0:128] == 0).all(), "top-left rect not zeroed"
    # Second rect: 0.80..0.95 of 720 = rows 576..684, cols 1024..1216
    assert (masked[576:684, 1024:1216] == 0).all(), "bottom-right rect not zeroed"
    print("✓ apply_hud_mask handles multiple rects")


def test_apply_hud_mask_empty_list_is_passthrough():
    f = make_frame()
    masked = apply_hud_mask(f, [])
    assert np.array_equal(f, masked), "Empty mask list should leave frame unchanged"
    print("✓ apply_hud_mask with empty list is a pass-through")


def test_apply_hud_mask_handles_out_of_bounds_coords():
    """Clamps normalized coords > 1.0 without crashing."""
    f = make_frame()
    masked = apply_hud_mask(f, [{"x": 0.9, "y": 0.9, "w": 0.5, "h": 0.5}])
    # Should still produce a same-shape image and zero the corner region
    assert masked.shape == f.shape
    assert (masked[-10:, -10:] == 0).all()
    print("✓ apply_hud_mask clamps out-of-bounds coords")


# ─── Non-gameplay filter ─────────────────────────────────────────────────

def test_non_gameplay_rejects_near_black():
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    reason = looks_like_non_gameplay(f)
    assert reason is not None, "Pure black frame should be rejected"
    print(f"✓ non-gameplay rejects black frame (reason={reason})")


def test_non_gameplay_rejects_near_white():
    f = np.full((720, 1280, 3), 250, dtype=np.uint8)
    reason = looks_like_non_gameplay(f)
    assert reason is not None, "Pure white frame should be rejected"
    print(f"✓ non-gameplay rejects white frame (reason={reason})")


def test_non_gameplay_rejects_flat_color():
    f = np.full((720, 1280, 3), 80, dtype=np.uint8)  # flat gray — no texture
    reason = looks_like_non_gameplay(f)
    assert reason is not None, "Flat-color frame should be rejected"
    print(f"✓ non-gameplay rejects flat-color frame (reason={reason})")


def test_non_gameplay_accepts_textured_frame():
    f = make_frame()
    reason = looks_like_non_gameplay(f)
    assert reason is None, f"Textured gameplay-like frame should pass, got reason={reason}"
    print("✓ non-gameplay accepts textured gameplay-like frame")


# ─── Entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_apply_hud_mask_zeros_the_rectangle()
    test_apply_hud_mask_multiple_rects()
    test_apply_hud_mask_empty_list_is_passthrough()
    test_apply_hud_mask_handles_out_of_bounds_coords()
    test_non_gameplay_rejects_near_black()
    test_non_gameplay_rejects_near_white()
    test_non_gameplay_rejects_flat_color()
    test_non_gameplay_accepts_textured_frame()
    print("\nAll video_ingester tests passed.")

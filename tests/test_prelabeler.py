"""
Pre-labeler tests. Skips if torch/ultralytics aren't installed,
so CI can run without the GPU stack.
Run with: python -m tests.test_prelabeler
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports_optional():
    """prelabeler module should be importable even without the heavy deps."""
    from backend import prelabeler
    assert hasattr(prelabeler, "prelabel_both")
    assert hasattr(prelabeler, "prelabel_segmentation")
    assert hasattr(prelabeler, "prelabel_detection")
    print("✓ imports_optional")


def test_constants():
    from backend import prelabeler
    # Sanity check the COCO class sets
    assert 2 in prelabeler.VEHICLE_CLS  # car
    assert 7 in prelabeler.VEHICLE_CLS  # truck
    assert 11 in prelabeler.SIGN_CLS    # stop sign
    print("✓ constants")


def test_segmentation_roundtrip_if_available():
    try:
        import torch  # noqa
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large  # noqa
    except ImportError:
        print("⚠ skipping seg roundtrip (torch/torchvision not installed)")
        return
    from backend.prelabeler import prelabel_segmentation
    frame = np.random.randint(30, 180, (360, 640, 3), dtype=np.uint8)
    out = prelabel_segmentation(frame)
    assert "mask_png_b64" in out
    assert "classes" in out
    assert len(out["classes"]) == 4
    print("✓ segmentation_roundtrip")


def test_detection_roundtrip_if_available():
    try:
        import ultralytics  # noqa
    except ImportError:
        print("⚠ skipping det roundtrip (ultralytics not installed)")
        return
    from backend.prelabeler import prelabel_detection
    frame = np.random.randint(30, 180, (360, 640, 3), dtype=np.uint8)
    out = prelabel_detection(frame)
    assert "boxes" in out
    assert isinstance(out["boxes"], list)
    print("✓ detection_roundtrip")


if __name__ == "__main__":
    test_imports_optional()
    test_constants()
    test_segmentation_roundtrip_if_available()
    test_detection_roundtrip_if_available()
    print("\nAll prelabeler tests passed (or skipped).")

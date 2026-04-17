"""
Tests for training/model.py.

Run with: python -m tests.test_model

If torch is not installed, these tests skip cleanly — on your machine with
torch installed, they run the real checks against the multi-task model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _skip(msg):
    print(f"⊘ skipped: {msg}")


def test_constants():
    """model.py exports the shape constants train.py and predict.py need."""
    from training.model import INPUT_H, INPUT_W, NUM_SEG_CLASSES, NUM_DET_CLASSES
    assert INPUT_H == 288
    assert INPUT_W == 512
    assert NUM_SEG_CLASSES == 4     # offroad, road, curb, wall
    assert NUM_DET_CLASSES == 2     # vehicle, sign
    print("✓ model constants are as expected")


def test_model_builds_and_forward_shape():
    try:
        import torch
    except ImportError:
        _skip("torch not installed"); return

    from training.model import PerceptionModel, INPUT_H, INPUT_W, NUM_SEG_CLASSES
    m = PerceptionModel()
    m.eval()
    x = torch.randn(2, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = m(x)
    assert "seg" in out and "det" in out, "Model output must contain seg + det heads"
    # Seg output should be (batch, classes, H, W) — possibly at lower res
    seg = out["seg"]
    assert seg.shape[0] == 2 and seg.shape[1] == NUM_SEG_CLASSES, \
        f"Bad seg shape: {tuple(seg.shape)}"
    # Det output is a list of multi-scale feature maps
    assert isinstance(out["det"], list) and len(out["det"]) >= 1
    print(f"✓ model builds, forward OK, seg shape {tuple(seg.shape)}, "
          f"{len(out['det'])} det scales")


def test_seg_loss_handles_all_ignored():
    """If every target pixel is 255 (ignore), the loss should still be finite."""
    try:
        import torch
    except ImportError:
        _skip("torch not installed"); return

    from training.model import seg_loss, NUM_SEG_CLASSES
    pred = torch.randn(1, NUM_SEG_CLASSES, 32, 32, requires_grad=True)
    target = torch.full((1, 32, 32), 255, dtype=torch.long)
    loss = seg_loss(pred, target)
    assert torch.isfinite(loss) or loss.item() == 0.0, \
        f"seg_loss on fully-ignored target should be finite, got {loss.item()}"
    print(f"✓ seg_loss handles fully-ignored target (loss={loss.item():.4f})")


def test_det_loss_handles_empty_targets():
    """Empty detection targets (no boxes) must not blow up the loss."""
    try:
        import torch
    except ImportError:
        _skip("torch not installed"); return

    from training.model import det_loss, PerceptionModel, INPUT_H, INPUT_W
    m = PerceptionModel(); m.eval()
    x = torch.randn(1, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = m(x)
    det_preds = out["det"]
    # Everything detached → rebuild tensors with grad so the loss can backprop
    det_preds = [p.clone().detach().requires_grad_(True) for p in det_preds]
    empty_targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]
    loss = det_loss(det_preds, empty_targets)
    assert torch.isfinite(loss), f"det_loss must be finite for empty targets, got {loss.item()}"
    print(f"✓ det_loss handles empty targets (loss={loss.item():.4f})")


def test_decode_segmentation_returns_class_indices():
    try:
        import torch
    except ImportError:
        _skip("torch not installed"); return

    from training.model import decode_segmentation, NUM_SEG_CLASSES
    logits = torch.randn(2, NUM_SEG_CLASSES, 32, 32)
    pred = decode_segmentation(logits)
    assert pred.dtype in (torch.int64, torch.long, torch.int32)
    assert pred.min().item() >= 0
    assert pred.max().item() < NUM_SEG_CLASSES
    print("✓ decode_segmentation returns valid class indices")


def test_frame_uncertainty_is_scalar():
    try:
        import torch
    except ImportError:
        _skip("torch not installed"); return

    from training.model import frame_uncertainty, PerceptionModel, INPUT_H, INPUT_W
    m = PerceptionModel(); m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, INPUT_H, INPUT_W))
    u = frame_uncertainty(out["seg"], out["det"])
    assert isinstance(u, float)
    assert 0.0 <= u <= 10.0, f"Expected reasonable uncertainty scalar, got {u}"
    print(f"✓ frame_uncertainty returns a reasonable scalar ({u:.3f})")


if __name__ == "__main__":
    test_constants()
    test_model_builds_and_forward_shape()
    test_seg_loss_handles_all_ignored()
    test_det_loss_handles_empty_targets()
    test_decode_segmentation_returns_class_indices()
    test_frame_uncertainty_is_scalar()
    print("\nAll model tests passed (skipped ones note torch was missing).")

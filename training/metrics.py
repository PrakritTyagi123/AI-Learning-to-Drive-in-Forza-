"""
ForzaTek AI — Evaluation Metrics
=================================
Shared metrics used by train.py during validation and by any future
stand-alone evaluation scripts.

All functions accept PyTorch tensors and return plain Python floats / lists,
so they can be logged to JSON without further conversion.
"""
from __future__ import annotations

from typing import List

import numpy as np


def compute_iou(pred, target, num_classes: int) -> List[float]:
    """Per-class IoU. Returns `nan` for classes absent from both pred and target.

    `pred` and `target` are integer tensors with the same shape.
    """
    ious: list[float] = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((inter / union).item())
    return ious


def mean_iou(pred, target, num_classes: int) -> float:
    ious = [i for i in compute_iou(pred, target, num_classes) if not np.isnan(i)]
    if not ious:
        return 0.0
    return float(np.mean(ious))


def compute_pixel_accuracy(pred, target, ignore_index: int = 255) -> float:
    """Fraction of valid pixels (those not equal to `ignore_index`) predicted correctly."""
    valid = (target != ignore_index)
    if valid.sum().item() == 0:
        return 0.0
    correct = ((pred == target) & valid).sum().item()
    return correct / valid.sum().item()


def _bbox_iou(a: list, b: list) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_detection_map(all_preds: list, all_gts: list, iou_thr: float = 0.5) -> float:
    """Simple mean Average Precision @ iou_thr.

    all_preds: list of lists of (box, score, cls) — one list per image
    all_gts:   list of lists of (box, cls)         — one list per image

    This is the "pragmatic" mAP used for monitoring training, not the full
    COCO-style 11-point interpolation.
    """
    classes = set()
    for preds in all_preds:
        for _, _, c in preds:
            classes.add(c)
    for gts in all_gts:
        for _, c in gts:
            classes.add(c)
    if not classes:
        return 0.0

    aps: list[float] = []
    for c in sorted(classes):
        tp_fp: list[tuple[float, int]] = []
        n_gt = 0

        for preds, gts in zip(all_preds, all_gts):
            gt_boxes = [b for b, gc in gts if gc == c]
            n_gt += len(gt_boxes)
            matched = [False] * len(gt_boxes)
            for box, score, pc in sorted(
                (p for p in preds if p[2] == c), key=lambda x: -x[1]
            ):
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    iou = _bbox_iou(box, gb)
                    if iou > best_iou:
                        best_iou = iou; best_j = j
                if best_iou >= iou_thr and best_j >= 0:
                    matched[best_j] = True
                    tp_fp.append((score, 1))
                else:
                    tp_fp.append((score, 0))

        if n_gt == 0 or not tp_fp:
            continue
        tp_fp.sort(key=lambda x: -x[0])
        tp_cum, fp_cum = 0, 0
        precisions: list[float] = []
        recalls: list[float] = []
        for _, is_tp in tp_fp:
            if is_tp: tp_cum += 1
            else:     fp_cum += 1
            precisions.append(tp_cum / max(1, tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)
        # Area under PR curve (trapezoidal)
        ap = 0.0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def build_confusion_matrix(pred, target, num_classes: int, ignore_index: int = 255):
    """Returns an NxN numpy int matrix. Rows = truth, cols = prediction."""
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    pred_np = pred.detach().cpu().numpy().ravel()
    tgt_np = target.detach().cpu().numpy().ravel()
    valid = tgt_np != ignore_index
    for p, t in zip(pred_np[valid], tgt_np[valid]):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1
    return mat

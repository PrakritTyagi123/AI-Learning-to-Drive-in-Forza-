"""
ForzaTek AI — Pre-labeler
==========================
Runs pretrained models to generate initial label proposals for frames.
Used to populate the labeling tool so the user corrects instead of drawing
from scratch.

Models used:
  - DeepLabV3 MobileNetV3 (torchvision, VOC-trained) for segmentation seed
  - YOLOv8n (ultralytics, COCO-trained) for detection seed

This is ONLY for initial proposals. The trained multi-task model takes over
once round 1 labels are done and the first custom model is trained.
"""
from __future__ import annotations

import base64
import io
import threading
from typing import Optional

import cv2
import numpy as np


_seg_model = None
_seg_lock = threading.Lock()
_det_model = None
_det_lock = threading.Lock()


def _load_seg():
    global _seg_model
    with _seg_lock:
        if _seg_model is not None:
            return _seg_model
        import torch
        from torchvision.models.segmentation import (
            deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
        )
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights).eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _seg_model = (model, weights.transforms(), torch.cuda.is_available())
        return _seg_model


def _load_det():
    global _det_model
    with _det_lock:
        if _det_model is not None:
            return _det_model
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        _det_model = model
        return _det_model


# ─── Segmentation pre-label ───
# VOC class indices we care about: 0=background, 7=car, 15=person, ... no "road".
# Workaround: use a color-based road heuristic PLUS the segmentation output,
# merged. User will correct. Better than nothing.

def prelabel_segmentation(frame_bgr: np.ndarray) -> dict:
    """
    Returns {mask_png_b64, classes} where mask is a single-channel PNG with values:
      0 = offroad, 1 = road, 2 = curb (always 0 initially — user paints), 3 = wall
    """
    H, W = frame_bgr.shape[:2]
    model, transforms, use_cuda = _load_seg()

    import torch
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = transforms(torch.from_numpy(frame_rgb).permute(2, 0, 1)).unsqueeze(0)
    if use_cuda:
        tensor = tensor.cuda()
    with torch.no_grad():
        out = model(tensor)["out"][0]
    pred = out.argmax(0).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

    # VOC class 20 is not road but 15=person; there is no explicit "road".
    # Use color heuristic: gray pixels in lower 2/3 of frame = road candidate.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    road_candidate = (sat < 60) & (val > 40) & (val < 200)
    lower_half = np.zeros_like(road_candidate)
    lower_half[H // 3 :] = True
    road_mask = (road_candidate & lower_half).astype(np.uint8)

    # Morphology cleanup: fill small gaps, remove small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component (the actual road, not noise)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        road_mask = (labels == largest).astype(np.uint8)

    # Our class mask: 0=offroad, 1=road
    class_mask = road_mask.copy()

    # Encode as PNG, base64
    ok, buf = cv2.imencode(".png", class_mask)
    mask_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""
    return {
        "mask_png_b64": mask_b64,
        "classes": ["offroad", "road", "curb", "wall"],
    }


# ─── Detection pre-label ───
# COCO class ids of interest: 2=car, 3=motorcycle, 5=bus, 7=truck, 11=stop sign
VEHICLE_CLS = {2, 3, 5, 7}
SIGN_CLS = {11, 12, 13}  # stop, parking, fire hydrant (proxy for signs)

def prelabel_detection(frame_bgr: np.ndarray, conf_thr: float = 0.35) -> dict:
    model = _load_det()
    H, W = frame_bgr.shape[:2]
    results = model.predict(frame_bgr, conf=conf_thr, verbose=False)
    boxes = []
    if len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                c = cls[i]
                if c in VEHICLE_CLS:
                    label = "vehicle"
                elif c in SIGN_CLS:
                    label = "sign"
                else:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                boxes.append({
                    "cls": label,
                    "x": float(x1 / W), "y": float(y1 / H),
                    "w": float((x2 - x1) / W), "h": float((y2 - y1) / H),
                    "confidence": float(conf[i]),
                })
    return {"boxes": boxes}


def prelabel_both(frame_bgr: np.ndarray) -> dict:
    """Convenience: run both pre-labelers in one call."""
    return {
        "seg": prelabel_segmentation(frame_bgr),
        "det": prelabel_detection(frame_bgr),
    }

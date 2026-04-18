"""
ForzaTek AI — Pre-labeler
==========================
Runs pretrained models to generate initial label proposals for frames.

Models used:
  - YOLO (ultralytics, COCO) for vehicle / sign detection
    Variant controlled by env YOLO_MODEL (default: yolo11x)
  - SegFormer-B5 fine-tuned on Cityscapes for road / sidewalk / wall / vegetation
    segmentation. Maps Cityscapes classes to our 4 classes:
      0 = offroad  (everything else: terrain, vegetation, sky, ...)
      1 = road     (road, motorcycle, bicycle)
      2 = curb     (sidewalk)
      3 = wall     (wall, fence, building)

Cityscapes class IDs:
  0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole, 6 traffic light,
  7 traffic sign, 8 vegetation, 9 terrain, 10 sky, 11 person, 12 rider,
  13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle
"""
from __future__ import annotations

import base64
import os
import threading

import cv2
import numpy as np


_seg_model = None
_seg_lock = threading.Lock()
_det_model = None
_det_lock = threading.Lock()


# ───────── YOLO loader ─────────

YOLO_MODEL_NAME = os.environ.get("YOLO_MODEL", "yolo11x")


def _load_det():
    """Lazy-load YOLO."""
    global _det_model
    with _det_lock:
        if _det_model is not None:
            return _det_model
        from ultralytics import YOLO
        print(f"[PRELABEL] Loading YOLO model: {YOLO_MODEL_NAME}.pt")
        model = YOLO(f"{YOLO_MODEL_NAME}.pt")
        try:
            import torch
            if torch.cuda.is_available():
                model.to("cuda")
                print("[PRELABEL] YOLO using CUDA")
            else:
                print("[PRELABEL] YOLO using CPU (slow!)")
        except Exception:
            pass
        _det_model = model
        return _det_model


# ───────── Segmentation loader — SegFormer-B5 Cityscapes ─────────

SEG_MODEL_NAME = os.environ.get(
    "SEG_MODEL", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
)


def _load_seg():
    """Load SegFormer from HuggingFace transformers."""
    global _seg_model
    with _seg_lock:
        if _seg_model is not None:
            return _seg_model
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        print(f"[PRELABEL] Loading SegFormer: {SEG_MODEL_NAME}")
        print("[PRELABEL] First run downloads ~320 MB of weights")
        processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME).eval()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()
            print("[PRELABEL] SegFormer using CUDA")
        else:
            print("[PRELABEL] SegFormer using CPU (slow!)")
        _seg_model = (model, processor, use_cuda)
        return _seg_model


# ───────── Cityscapes → ForzaTek class mapping ─────────
# 0 offroad | 1 road | 2 curb | 3 wall
CITYSCAPES_TO_OURS = {
    0: 1,   # road -> road
    1: 2,   # sidewalk -> curb
    # everything else stays 0 (offroad)
}


def prelabel_segmentation(frame_bgr: np.ndarray) -> dict:
    """
    Returns {mask_png_b64, classes} where mask is single-channel uint8:
      0 = offroad, 1 = road, 2 = curb, 3 = wall
    """
    H, W = frame_bgr.shape[:2]
    model, processor, use_cuda = _load_seg()

    import torch
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    if use_cuda:
        pixel_values = pixel_values.cuda()

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, 19, H/4, W/4]

    # Upsample to original resolution
    upsampled = torch.nn.functional.interpolate(
        logits, size=(H, W), mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Map Cityscapes classes to our 4 classes
    class_mask = np.zeros((H, W), dtype=np.uint8)
    for cs_id, our_id in CITYSCAPES_TO_OURS.items():
        class_mask[pred == cs_id] = our_id

    # Player car fix: the player's car always sits in the center-bottom region.
    # Cityscapes will classify it as "car" (not road), creating a hole in the
    # road mask. Fill in car pixels that fall inside the player-car region
    # as road — this is where the player car actually is, and the underlying
    # asphalt should be treated as drivable.
    player_region = np.zeros((H, W), dtype=bool)
    # Region: middle 40% horizontally, bottom 55% vertically
    player_region[int(H * 0.45):, int(W * 0.30):int(W * 0.70)] = True
    # Cityscapes classes for the player's vehicle: 13 car, 14 truck, 15 bus, 17 motorcycle
    player_car_mask = np.isin(pred, [13, 14, 15, 17]) & player_region
    # Only fill in connected-to-road pixels (morph close bridges the car to adjacent road)
    combined = ((class_mask == 1) | player_car_mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    filled = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    # Only use the filled version where the player region overlaps — don't
    # accidentally merge road into buildings on the side of the frame.
    fill_zone = np.zeros((H, W), dtype=bool)
    fill_zone[int(H * 0.40):, int(W * 0.20):int(W * 0.80)] = True
    class_mask[(filled == 1) & fill_zone] = 1

    ok, buf = cv2.imencode(".png", class_mask)
    mask_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""
    return {
        "mask_png_b64": mask_b64,
        "classes": ["offroad", "road", "curb", "wall"],
    }


# ───────── Detection pre-label ─────────

COCO_VEHICLE_CLS = {2, 3, 5, 7}   # car, motorcycle, bus, truck
COCO_SIGN_CLS    = {11, 12, 13}   # stop sign, parking meter, fire hydrant


def prelabel_detection(frame_bgr: np.ndarray, conf_thr: float = 0.25) -> dict:
    """YOLO vehicle/sign detection. Excludes the player's own car (center-bottom region)."""
    model = _load_det()
    H, W = frame_bgr.shape[:2]
    results = model.predict(frame_bgr, conf=conf_thr, imgsz=1280, verbose=False)

    # Player car region: box that always contains the player's vehicle
    # center 40% wide, bottom 55% tall
    p_x1, p_x2 = 0.30, 0.70
    p_y1, p_y2 = 0.45, 1.00

    boxes = []
    if len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                c = cls[i]
                if c in COCO_VEHICLE_CLS:
                    label = "vehicle"
                elif c in COCO_SIGN_CLS:
                    label = "sign"
                else:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                # Normalized center
                cx = ((x1 + x2) / 2.0) / W
                cy = ((y1 + y2) / 2.0) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H

                # Skip if it's the player's car:
                # - label is vehicle
                # - center is inside the player region
                # - box is BIG (player car is always large relative to frame)
                if label == "vehicle" and p_x1 <= cx <= p_x2 and p_y1 <= cy <= p_y2:
                    if bw > 0.15 or bh > 0.15:  # at least 15% of frame in some dim
                        continue  # skip — this is the player's car

                boxes.append({
                    "cls": label,
                    "x": float(x1 / W),
                    "y": float(y1 / H),
                    "w": float((x2 - x1) / W),
                    "h": float((y2 - y1) / H),
                    "confidence": float(conf[i]),
                })
    return {"boxes": boxes}


# ───────── Trained model loader (uses YOUR model if one is active) ─────────

_trained_model = None
_trained_path = None
_trained_lock = threading.Lock()


def _load_trained_model():
    """Load the user's active trained checkpoint, if any. Returns None if no active model."""
    global _trained_model, _trained_path
    with _trained_lock:
        # Check active model from DB
        try:
            from backend.database import get_active_model
            active = get_active_model()
        except Exception:
            return None
        if not active:
            _trained_model = None
            _trained_path = None
            return None

        ckpt_path = active.get("path")
        if not ckpt_path:
            return None

        from pathlib import Path as _Path
        if not _Path(ckpt_path).is_absolute():
            # Resolve relative to project root
            root = _Path(__file__).resolve().parent.parent
            ckpt_path = str(root / ckpt_path)

        # Reload if the active path changed
        if _trained_model is not None and _trained_path == ckpt_path:
            return _trained_model

        try:
            import torch
            from training.model import PerceptionModel
            print(f"[PRELABEL] Loading trained model from {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            # Support both full-state dicts and raw weights
            if isinstance(state, dict) and "model" in state:
                weights = state["model"]
            elif isinstance(state, dict) and "state_dict" in state:
                weights = state["state_dict"]
            else:
                weights = state
            model = PerceptionModel()
            model.load_state_dict(weights, strict=False)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                print("[PRELABEL] Trained model using CUDA")
            _trained_model = model
            _trained_path = ckpt_path
            return _trained_model
        except Exception as e:
            print(f"[PRELABEL] Failed to load trained model: {e}")
            _trained_model = None
            _trained_path = None
            return None


def prelabel_with_trained(frame_bgr: np.ndarray) -> dict | None:
    """Run the user's trained model. Returns None if no active model."""
    model = _load_trained_model()
    if model is None:
        return None

    import torch
    from training.model import decode_segmentation, decode_detection

    H, W = frame_bgr.shape[:2]
    # Get the training input size from the model module (single source of truth)
    try:
        from training.model import INPUT_H, INPUT_W
    except ImportError:
        INPUT_H, INPUT_W = 288, 512
    resized = cv2.resize(frame_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    use_cuda = next(model.parameters()).is_cuda
    if use_cuda:
        tensor = tensor.cuda()

    with torch.no_grad():
        out = model(tensor)

    # Seg: upsample to original resolution
    seg_pred = decode_segmentation(out["seg"])[0].cpu().numpy().astype(np.uint8)
    seg_resized = cv2.resize(seg_pred, (W, H), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", seg_resized)
    mask_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""

    # Det: decode boxes
    try:
        dets = decode_detection(out["det"], conf_thr=0.25)[0]
    except Exception as e:
        print(f"[PRELABEL] Det decode failed: {e}")
        dets = []

    # Scale boxes back to original image coords (model output is in model-input pixel coords)
    sx = W / INPUT_W
    sy = H / INPUT_H
    boxes = []
    CLS_NAMES = {0: "vehicle", 1: "sign"}
    for d in dets:
        x1, y1, x2, y2 = d["x1"] * sx, d["y1"] * sy, d["x2"] * sx, d["y2"] * sy
        cls_id = int(d.get("cls", 0))
        boxes.append({
            "cls": CLS_NAMES.get(cls_id, "vehicle"),
            "x": float(max(0, x1) / W),
            "y": float(max(0, y1) / H),
            "w": float((x2 - x1) / W),
            "h": float((y2 - y1) / H),
            "confidence": float(d.get("score", 0.0)),
        })

    return {
        "seg": {"mask_png_b64": mask_b64, "classes": ["offroad", "road", "curb", "wall"]},
        "det": {"boxes": boxes},
    }


def prelabel_both(frame_bgr: np.ndarray) -> dict:
    """Uses trained model if one is active, otherwise falls back to SegFormer + YOLO."""
    trained = prelabel_with_trained(frame_bgr)
    if trained is not None:
        return trained
    return {
        "seg": prelabel_segmentation(frame_bgr),
        "det": prelabel_detection(frame_bgr),
    }
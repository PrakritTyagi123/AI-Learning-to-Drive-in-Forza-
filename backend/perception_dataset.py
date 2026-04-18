"""
ForzaTek AI — Perception Training Dataset
==========================================
Pulls labeled frames from the SQLite database and yields tensors ready
for training the perception model.

- Segmentation labels are 4-class PNG masks (offroad, road, curb, wall)
  which we collapse to 2 classes (offroad=0, road=1) by mapping any non-1
  value to 0.
- Detection labels are {boxes: [{cls, x, y, w, h}]} in normalized 0-1 coords.
  We map "vehicle"->0, "sign"->1 and scale to pixel coords of the model input.
"""
from __future__ import annotations

import base64
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from backend.database import DB_PATH, read_conn
from backend.perception import INPUT_W, INPUT_H


CLASS_MAP = {"vehicle": 0, "sign": 1}


def _fetch_labeled_frame_ids() -> list[int]:
    """Return frame IDs that have BOTH seg and det labels."""
    with read_conn(DB_PATH) as c:
        rows = c.execute("""
            SELECT f.id
            FROM frames f
            WHERE f.label_status = 'labeled'
              AND EXISTS (SELECT 1 FROM labels l WHERE l.frame_id=f.id AND l.task='seg')
              AND EXISTS (SELECT 1 FROM labels l WHERE l.frame_id=f.id AND l.task='det')
        """).fetchall()
    return [r["id"] for r in rows]


def _load_frame(frame_id: int):
    """Load a frame's JPEG + seg PNG + det JSON from DB."""
    with read_conn(DB_PATH) as c:
        row = c.execute(
            "SELECT frame_jpeg, width, height FROM frames WHERE id=?",
            (frame_id,),
        ).fetchone()
        labels = c.execute(
            "SELECT task, data_json FROM labels WHERE frame_id=?",
            (frame_id,),
        ).fetchall()

    jpg = bytes(row["frame_jpeg"])
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    orig_h, orig_w = img.shape[:2]

    seg_mask = None
    det_boxes = []
    for l in labels:
        task = l["task"]
        data = json.loads(l["data_json"])
        if task == "seg":
            mask_b64 = data.get("mask_png_b64", "")
            if mask_b64:
                m = base64.b64decode(mask_b64)
                ma = np.frombuffer(m, dtype=np.uint8)
                mask = cv2.imdecode(ma, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    # Force single-channel (some PNGs come through as RGBA)
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                    # Collapse 4 classes to 2: only class 1 (road) stays 1, else 0
                    bin_mask = (mask == 1).astype(np.uint8)
                    seg_mask = bin_mask
        elif task == "det":
            for b in data.get("boxes", []):
                cls_name = b.get("cls", "vehicle")
                cls_id = CLASS_MAP.get(cls_name)
                if cls_id is None:
                    continue
                det_boxes.append({
                    "cls": cls_id,
                    "x": float(b.get("x", 0)),
                    "y": float(b.get("y", 0)),
                    "w": float(b.get("w", 0)),
                    "h": float(b.get("h", 0)),
                })

    return img, seg_mask, det_boxes, orig_w, orig_h


def _augment(img_bgr, seg_mask, det_boxes, training: bool):
    """Basic augmentation: horizontal flip, brightness/contrast, small hue jitter."""
    H, W = img_bgr.shape[:2]
    if training:
        # Horizontal flip ~50%
        if random.random() < 0.5:
            img_bgr = img_bgr[:, ::-1].copy()
            if seg_mask is not None:
                seg_mask = seg_mask[:, ::-1].copy()
            # Flip boxes (normalized coords: x' = 1 - x - w)
            for b in det_boxes:
                b["x"] = 1.0 - b["x"] - b["w"]

        # Brightness / contrast jitter
        if random.random() < 0.7:
            alpha = 1.0 + (random.random() - 0.5) * 0.4   # 0.8 - 1.2
            beta = (random.random() - 0.5) * 30            # -15 .. +15
            img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Hue jitter
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
            img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img_bgr, seg_mask, det_boxes


class PerceptionDataset(Dataset):
    def __init__(self, frame_ids: list[int], training: bool = True):
        self.ids = frame_ids
        self.training = training

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        img_bgr, seg_mask, det_boxes, _, _ = _load_frame(fid)

        # Resize image/mask to model input size
        img_rs = cv2.resize(img_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        if seg_mask is not None:
            seg_rs = cv2.resize(seg_mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
        else:
            seg_rs = np.full((INPUT_H, INPUT_W), 255, dtype=np.uint8)  # ignore

        # Augment
        img_rs, seg_rs, det_boxes = _augment(img_rs, seg_rs, det_boxes, self.training)

        # Convert to tensors
        img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1)  # (3, H, W)
        seg_t = torch.from_numpy(seg_rs).long()              # (H, W) values 0/1/255

        # Det boxes → pixel coords in model input space
        boxes = []
        labels = []
        for b in det_boxes:
            x1 = b["x"] * INPUT_W
            y1 = b["y"] * INPUT_H
            x2 = (b["x"] + b["w"]) * INPUT_W
            y2 = (b["y"] + b["h"]) * INPUT_H
            # Clip + skip tiny
            x1, x2 = max(0, x1), min(INPUT_W, x2)
            y1, y2 = max(0, y1), min(INPUT_H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(b["cls"])

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros((0,), dtype=torch.long),
        }
        return img_t, seg_t, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    segs = torch.stack([b[1] for b in batch], dim=0)
    targets = [b[2] for b in batch]
    return imgs, segs, targets


def make_splits(val_ratio: float = 0.15, seed: int = 42) -> tuple[list[int], list[int]]:
    ids = _fetch_labeled_frame_ids()
    if not ids:
        raise RuntimeError("No labeled frames found in DB. Label some frames first.")
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * (1.0 - val_ratio))
    return ids[:cut], ids[cut:]
"""
ForzaTek AI — Labeled Frames Dataset
=====================================
PyTorch-style dataset that pulls labeled frames from the SQLite database,
decodes images + segmentation masks + bounding boxes into tensors, and
applies augmentation.

Separated from train.py so that predict.py and the tests can reuse the
same loading logic without pulling in the training loop.
"""
from __future__ import annotations

import base64
import json
import random
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import DB_PATH, read_conn
from training.model import INPUT_H, INPUT_W


class LabeledFramesDataset:
    """Lazy, index-addressable dataset of (image, seg_mask, det_targets) tuples.

    Reads from the DB on every __getitem__ call — avoids holding 500+ images
    in RAM and lets PyTorch's DataLoader workers parallelize decoding.
    """

    def __init__(self, frame_ids: list[int], augment: bool = True):
        self.frame_ids = frame_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int):
        import torch

        fid = self.frame_ids[idx]
        with read_conn(DB_PATH) as c:
            f = c.execute(
                "SELECT frame_jpeg, width, height FROM frames WHERE id=?", (fid,)
            ).fetchone()
            labels = c.execute(
                "SELECT task, data_json FROM labels WHERE frame_id=?", (fid,)
            ).fetchall()

        # Decode JPEG
        arr = np.frombuffer(bytes(f["frame_jpeg"]), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Seg mask (255 = ignore)
        seg_mask = np.full((INPUT_H, INPUT_W), 255, dtype=np.uint8)
        boxes: list[list[float]] = []
        cls_ids: list[int] = []

        for row in labels:
            task = row["task"]
            data = json.loads(row["data_json"])
            if task == "seg":
                mask_b64 = data.get("mask_png_b64", "")
                if mask_b64:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
                    mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(mask, (INPUT_W, INPUT_H),
                                          interpolation=cv2.INTER_NEAREST)
                        seg_mask = mask
            elif task == "det":
                for b in data.get("boxes", []):
                    cls_name = b.get("cls", "vehicle")
                    cls_id = 0 if cls_name == "vehicle" else 1
                    x1 = b["x"] * INPUT_W
                    y1 = b["y"] * INPUT_H
                    x2 = (b["x"] + b["w"]) * INPUT_W
                    y2 = (b["y"] + b["h"]) * INPUT_H
                    boxes.append([x1, y1, x2, y2])
                    cls_ids.append(cls_id)

        # Augmentation: random horizontal flip + brightness jitter
        if self.augment and random.random() < 0.5:
            img_rgb = img_rgb[:, ::-1, :].copy()
            seg_mask = seg_mask[:, ::-1].copy()
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                boxes[i] = [INPUT_W - x2, y1, INPUT_W - x1, y2]
        if self.augment and random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img_rgb = np.clip(img_rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # To tensors + ImageNet normalization
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        seg_t = torch.from_numpy(seg_mask).long()
        det_t = {
            "boxes":  torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(cls_ids, dtype=torch.long)   if cls_ids else torch.zeros((0,), dtype=torch.long),
        }
        return img_t, seg_t, det_t


def collate(batch):
    """Default collate for our DataLoader — stacks images and segs, keeps dets as a list."""
    import torch
    imgs = torch.stack([b[0] for b in batch])
    segs = torch.stack([b[1] for b in batch])
    dets = [b[2] for b in batch]
    return imgs, segs, dets


def get_frame_splits(val_frac: float = 0.15, seed: int = 42) -> Tuple[list[int], list[int]]:
    """Return (train_ids, val_ids) stratified by game_version.

    Only considers frames with BOTH seg and det labels so the training loop
    always has a valid target for both heads.
    """
    with read_conn(DB_PATH) as c:
        rows = c.execute("""
            SELECT DISTINCT f.id, f.game_version
            FROM frames f
            WHERE f.id IN (
                SELECT frame_id FROM labels WHERE task='seg'
                INTERSECT
                SELECT frame_id FROM labels WHERE task='det'
            )
        """).fetchall()

    by_ver: dict[str, list[int]] = {}
    for r in rows:
        by_ver.setdefault(r["game_version"] or "unknown", []).append(r["id"])

    train_ids: list[int] = []
    val_ids: list[int] = []
    rng = random.Random(seed)
    for ver, ids in by_ver.items():
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_frac))
        val_ids  += ids[:n_val]
        train_ids += ids[n_val:]
    return train_ids, val_ids

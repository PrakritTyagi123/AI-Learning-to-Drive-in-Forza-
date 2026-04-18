"""
ForzaTek AI — Perception Runtime Inference
============================================
Fast wrapper around PerceptionModelV2 for real-time use (PPO driving loop).

Usage:
    from backend.perception_infer import PerceptionRuntime
    rt = PerceptionRuntime("models/perception_v1.pt")
    result = rt.infer(bgr_frame)  # dict with 'road_mask', 'boxes', 'features'
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from backend.perception import (
    PerceptionModelV2, decode_segmentation, decode_detection,
    INPUT_W, INPUT_H, DET_CLASS_NAMES,
)


class PerceptionRuntime:
    def __init__(self, ckpt_path: str | Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PerceptionModelV2(pretrained=False)
        state = torch.load(ckpt_path, map_location=self.device)
        weights = state.get("model", state)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval().to(self.device)

        # Try to use channels-last + half for speed on supported devices
        self.use_half = (self.device == "cuda")
        if self.use_half:
            self.model = self.model.half()

        # Warm up
        dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=self.device,
                            dtype=torch.float16 if self.use_half else torch.float32)
        with torch.no_grad():
            for _ in range(2):
                self.model(dummy)

        print(f"[PERCEPT] Loaded from {ckpt_path} on {self.device} (half={self.use_half})")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, conf_thr: float = 0.3):
        H_orig, W_orig = frame_bgr.shape[:2]
        # Resize + normalize
        rs = cv2.resize(frame_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.use_half:
            tensor = tensor.half()

        out = self.model(tensor)
        seg = decode_segmentation(out["seg"])[0].cpu().numpy().astype(np.uint8)
        # Upsample mask to original frame resolution
        road_mask = cv2.resize(seg, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

        dets = decode_detection(out["det"], conf_thr=conf_thr)[0]
        # Scale boxes back to original image coords
        sx = W_orig / INPUT_W
        sy = H_orig / INPUT_H
        boxes = []
        for d in dets:
            boxes.append({
                "x1": float(d["x1"] * sx), "y1": float(d["y1"] * sy),
                "x2": float(d["x2"] * sx), "y2": float(d["y2"] * sy),
                "score": float(d["score"]),
                "cls": int(d["cls"]),
                "cls_name": DET_CLASS_NAMES[int(d["cls"])] if int(d["cls"]) < len(DET_CLASS_NAMES) else "unk",
            })

        return {
            "road_mask": road_mask,   # uint8 (H, W) — 1 where road, 0 elsewhere
            "boxes": boxes,
            # Optional: features usable as PPO state (flattened last-stage features)
        }

    @torch.no_grad()
    def infer_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns the stride-32 feature tensor pooled to a fixed-size vector.
        Useful as a direct state input to the PPO policy network."""
        rs = cv2.resize(frame_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.use_half:
            tensor = tensor.half()
        _, _, f32 = self.model.backbone(tensor)
        feat = torch.nn.functional.adaptive_avg_pool2d(f32, 1).flatten(1)
        return feat.float().cpu().numpy().squeeze(0)
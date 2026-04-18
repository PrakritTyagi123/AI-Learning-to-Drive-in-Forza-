"""
ForzaTek AI — Multi-Task Perception Model
==========================================
Shared EfficientNet-B1 backbone with two heads:
  - Segmentation head: 4-class semantic segmentation (offroad, road, curb, wall)
  - Detection head:    anchor-based object detection (vehicle, sign)

Lane head is stubbed for now (will be wired later).

Design choices explained
------------------------
- Backbone: EfficientNet-B1 (timm). Good accuracy-to-speed ratio. Pretrained
  on ImageNet so we start with useful low-level features even when our
  custom dataset is small.
- Segmentation decoder: lightweight FPN-style upsampler. Takes multi-scale
  features from the backbone, fuses them, upsamples to input resolution.
- Detection head: single-stage, 3-scale prediction (P3/P4/P5 FPN levels).
  Each cell predicts (objectness, x, y, w, h, class_logits). Simple, no
  anchors to tune.
- Input resolution: 512 x 288 (16:9). Matches Forza aspect ratio after HUD
  crop. Small enough for fast training, big enough to see distant cars.

The model file intentionally avoids anything too fancy (no transformers,
no attention, no fancy loss tricks) because this needs to train on 3000
labeled frames. Complexity should go into the DATA pipeline, not the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_H = 288
INPUT_W = 512

NUM_SEG_CLASSES = 4   # offroad, road, curb, wall
NUM_DET_CLASSES = 2   # vehicle, sign


# ─── Backbone ───

def build_backbone():
    """EfficientNet-B1 from timm, with feature extraction enabled."""
    import timm
    backbone = timm.create_model(
        "efficientnet_b1",
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4),  # strides 4, 8, 16, 32
    )
    return backbone


# ─── Segmentation head: lightweight FPN decoder ───

class SegHead(nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int = NUM_SEG_CLASSES):
        super().__init__()
        # in_channels = [c_s4, c_s8, c_s16, c_s32]
        c = 128
        self.lat = nn.ModuleList([nn.Conv2d(ic, c, 1) for ic in in_channels])
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ) for _ in in_channels
        ])
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # Top-down pathway with nearest upsampling
        lat = [l(f) for l, f in zip(self.lat, feats)]
        x = lat[-1]
        x = self.fuse[-1](x)
        for i in range(len(lat) - 2, -1, -1):
            x = F.interpolate(x, size=lat[i].shape[-2:], mode="nearest")
            x = x + lat[i]
            x = self.fuse[i](x)
        # Upsample to input resolution
        x = F.interpolate(x, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False)
        return self.out(x)


# ─── Detection head ───
# Simple anchor-free FCOS-lite: per-cell predict (objectness, lrtb, cls)

class DetHead(nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int = NUM_DET_CLASSES):
        super().__init__()
        c = 128
        self.num_classes = num_classes
        # Use the last 3 scales (8, 16, 32) as detection levels
        self.det_scales = 3
        self.lat = nn.ModuleList([nn.Conv2d(ic, c, 1) for ic in in_channels[-3:]])
        self.tower = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        # Outputs per cell: 1 (obj) + 4 (l,r,t,b) + num_classes
        self.out = nn.Conv2d(c, 1 + 4 + num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for l, f in zip(self.lat, feats[-3:]):
            x = l(f)
            x = self.tower(x)
            x = self.out(x)
            out.append(x)
        return out  # list of 3 tensors at different scales


# ─── Full model ───

class PerceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_backbone()
        # Query feature channel dims dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
            feats = self.backbone(dummy)
            in_channels = [f.shape[1] for f in feats]
        self.seg_head = SegHead(in_channels)
        self.det_head = DetHead(in_channels)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone(x)
        return {
            "seg": self.seg_head(feats),
            "det": self.det_head(feats),
        }


# ─── Loss functions ───

def seg_loss(pred: torch.Tensor, target: torch.Tensor,
             ignore_index: int = 255) -> torch.Tensor:
    """
    Standard cross-entropy with optional class weighting.
    Target is (B, H, W) long with class indices 0..num_classes-1 and 255=ignore.
    """
    return F.cross_entropy(pred, target, ignore_index=ignore_index)


def det_loss(preds: list[torch.Tensor], targets: list[dict]) -> torch.Tensor:
    """
    Simplified detection loss.
    preds: 3 feature maps, each (B, 1+4+NC, H, W)
    targets: list of length B, each is {'boxes': tensor(N,4), 'labels': tensor(N,)}
    Boxes are (x1, y1, x2, y2) in image pixel coords.

    We compute:
      - objectness BCE on all cells whose center falls inside a GT box
      - regression L1 loss on cells matched to a GT
      - classification CE on matched cells
    This is a teaching-level implementation — not SOTA, but stable and trainable.
    """
    device = preds[0].device
    B = preds[0].shape[0]
    total_obj = torch.tensor(0.0, device=device)
    total_reg = torch.tensor(0.0, device=device)
    total_cls = torch.tensor(0.0, device=device)
    pos_count = 0

    strides = [8, 16, 32]
    for pred, stride in zip(preds, strides):
        Bp, C, Hp, Wp = pred.shape
        # Split channels
        obj = pred[:, 0]
        reg = pred[:, 1:5]       # (B, 4, H, W) — l, r, t, b
        cls = pred[:, 5:]        # (B, NC, H, W)

        # Build target tensors — always float32 (targets shouldn't be half)
        obj_tgt = torch.zeros_like(obj, dtype=torch.float32)
        reg_tgt = torch.zeros_like(reg, dtype=torch.float32)
        cls_tgt = torch.full((Bp, Hp, Wp), -1, dtype=torch.long, device=device)

        # Grid of cell centers in pixel space
        ys = (torch.arange(Hp, device=device).float() + 0.5) * stride
        xs = (torch.arange(Wp, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        for b in range(Bp):
            tgt = targets[b]
            boxes = tgt["boxes"]
            labels = tgt["labels"]
            if len(boxes) == 0:
                continue
            for bi in range(len(boxes)):
                x1, y1, x2, y2 = boxes[bi]
                inside = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
                if inside.sum() == 0:
                    continue
                obj_tgt[b][inside] = 1.0
                reg_tgt[b, 0][inside] = (grid_x[inside] - x1) / stride
                reg_tgt[b, 1][inside] = (x2 - grid_x[inside]) / stride
                reg_tgt[b, 2][inside] = (grid_y[inside] - y1) / stride
                reg_tgt[b, 3][inside] = (y2 - grid_y[inside]) / stride
                cls_tgt[b][inside] = labels[bi]

        # Cast preds to float32 to match float32 targets (AMP safety)
        obj_f = obj.float()
        reg_f = reg.float()
        cls_f = cls.float()

        # Objectness loss on all cells
        total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_f, obj_tgt)

        # Reg + cls only on positive cells
        pos_mask = obj_tgt > 0.5
        if pos_mask.sum() > 0:
            total_reg = total_reg + F.l1_loss(
                reg_f.permute(0, 2, 3, 1)[pos_mask],
                reg_tgt.permute(0, 2, 3, 1)[pos_mask],
            )
            valid = cls_tgt >= 0
            if valid.sum() > 0:
                cls_flat = cls_f.permute(0, 2, 3, 1)[valid]
                tgt_flat = cls_tgt[valid]
                total_cls = total_cls + F.cross_entropy(cls_flat, tgt_flat)
            pos_count += int(pos_mask.sum().item())

    return total_obj + 1.0 * total_reg + 0.5 * total_cls


# ─── Inference decoding ───

@torch.no_grad()
def decode_segmentation(seg_logits: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) logits -> (B, H, W) class indices"""
    return seg_logits.argmax(dim=1)


@torch.no_grad()
def decode_detection(preds: list[torch.Tensor], conf_thr: float = 0.3,
                     iou_thr: float = 0.5, conf_threshold: float = None,
                     iou_threshold: float = None) -> list[list[dict]]:
    # Accept both old and new parameter names
    if conf_threshold is not None: conf_thr = conf_threshold
    if iou_threshold is not None: iou_thr = iou_threshold
    """
    Returns list of length B, each a list of {x1,y1,x2,y2,score,cls} in pixel coords.
    """
    from torchvision.ops import nms
    strides = [8, 16, 32]
    B = preds[0].shape[0]
    out_per_img = [[] for _ in range(B)]
    for pred, stride in zip(preds, strides):
        Bp, C, Hp, Wp = pred.shape
        obj = torch.sigmoid(pred[:, 0])
        reg = pred[:, 1:5]
        cls = pred[:, 5:].softmax(dim=1)

        ys = (torch.arange(Hp, device=pred.device).float() + 0.5) * stride
        xs = (torch.arange(Wp, device=pred.device).float() + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        for b in range(Bp):
            score = obj[b]
            mask = score > conf_thr
            if mask.sum() == 0:
                continue
            l = reg[b, 0][mask] * stride
            r = reg[b, 1][mask] * stride
            t = reg[b, 2][mask] * stride
            bo = reg[b, 3][mask] * stride
            cx = gx[mask]
            cy = gy[mask]
            x1 = cx - l; x2 = cx + r; y1 = cy - t; y2 = cy + bo
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            scores = score[mask]
            cls_probs = cls[b].permute(1, 2, 0)[mask]
            cls_id = cls_probs.argmax(dim=1)
            final_score = scores * cls_probs.max(dim=1).values
            keep = nms(boxes, final_score, iou_thr)
            for k in keep.tolist():
                out_per_img[b].append({
                    "x1": float(boxes[k, 0]),
                    "y1": float(boxes[k, 1]),
                    "x2": float(boxes[k, 2]),
                    "y2": float(boxes[k, 3]),
                    "score": float(final_score[k]),
                    "cls": int(cls_id[k]),
                })
    return out_per_img


# ─── Uncertainty computation (for active learning) ───

@torch.no_grad()
def frame_uncertainty(seg_logits: torch.Tensor, det_preds: list[torch.Tensor]) -> float:
    """
    Single scalar: higher = less confident.

    Segmentation uncertainty: mean entropy across pixels.
    Detection uncertainty: 1 minus average top-box confidence.
    Combine with equal weight.
    """
    probs = F.softmax(seg_logits, dim=1).clamp(min=1e-6)
    entropy = -(probs * probs.log()).sum(dim=1).mean().item()
    # Normalize entropy to roughly 0..1 (log(C) is the max)
    max_entropy = torch.log(torch.tensor(float(probs.shape[1]))).item()
    seg_unc = entropy / max_entropy

    det_confs = []
    for p in det_preds:
        obj = torch.sigmoid(p[:, 0])
        det_confs.append(obj.max().item() if obj.numel() else 0.0)
    det_unc = 1.0 - (sum(det_confs) / max(1, len(det_confs)))

    return 0.5 * seg_unc + 0.5 * det_unc
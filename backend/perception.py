"""
ForzaTek AI — Runtime Perception Model
========================================
Small, fast multitask CNN that outputs:
  - 2-class segmentation: road vs offroad
  - Bounding-box detection: vehicles and signs

Designed for 60+ FPS inference on an RTX 4080 so it can feed the PPO
driving policy in real time.

Input:  RGB frame, 256x144 (W x H)
Output:
  seg_logits: (B, 2, 144, 256)
  det_preds:  list of 3 tensors (strides 8, 16, 32), each (B, C, h, w)
              where C = 1 + 4 + NUM_DET_CLASSES = 7
              channels: [objectness, l, t, r, b, cls0_vehicle, cls1_sign]

Backbone: MobileNetV3-Small (pretrained ImageNet)
Params: ~3M
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ─── Configuration (single source of truth) ───
INPUT_W = 256
INPUT_H = 144
NUM_SEG_CLASSES = 2      # 0 = offroad, 1 = road
NUM_DET_CLASSES = 2      # 0 = vehicle, 1 = sign
DET_CLASS_NAMES = ["vehicle", "sign"]
SEG_CLASS_NAMES = ["offroad", "road"]
STRIDES = (8, 16, 32)


# ─── Backbone wrapper that exposes multi-scale features ───

class MobileBackbone(nn.Module):
    """MobileNetV3-Small that returns features at strides 8, 16, 32.

    The native feature extractor outputs 576-dim features at stride 32.
    We hook into intermediate layers to also get stride-8 and stride-16 features.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        # MobileNetV3-Small has 13 blocks in features
        # After block 2 → stride 8, 24 ch
        # After block 7 → stride 16, 48 ch
        # After block 12 → stride 32, 576 ch (with the last 1x1 conv)
        self.stage1 = nn.Sequential(*m.features[0:3])     # stride 8, 24 ch
        self.stage2 = nn.Sequential(*m.features[3:8])     # stride 16, 48 ch
        self.stage3 = nn.Sequential(*m.features[8:])      # stride 32, 576 ch
        self.ch_s8 = 24
        self.ch_s16 = 48
        self.ch_s32 = 576

    def forward(self, x):
        f8 = self.stage1(x)
        f16 = self.stage2(f8)
        f32 = self.stage3(f16)
        return f8, f16, f32


# ─── Segmentation head — lightweight FPN decoder ───

class SegHead(nn.Module):
    def __init__(self, ch8: int, ch16: int, ch32: int, mid: int = 64,
                 num_classes: int = NUM_SEG_CLASSES):
        super().__init__()
        self.reduce8  = nn.Conv2d(ch8,  mid, 1)
        self.reduce16 = nn.Conv2d(ch16, mid, 1)
        self.reduce32 = nn.Conv2d(ch32, mid, 1)
        self.smooth1 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(mid, num_classes, 1)

    def forward(self, f8, f16, f32, out_size):
        p32 = self.reduce32(f32)
        p16 = self.reduce16(f16) + F.interpolate(p32, size=f16.shape[-2:], mode="bilinear", align_corners=False)
        p16 = self.smooth1(p16)
        p8 = self.reduce8(f8) + F.interpolate(p16, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.smooth2(p8)
        logits = self.classifier(p8)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


# ─── Detection head — anchor-free FCOS-style (one per stride) ───

class DetHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = NUM_DET_CLASSES, mid: int = 64):
        super().__init__()
        # Shared towers
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        # Output: [obj, l, t, r, b, cls_logits_0..N-1]
        self.pred = nn.Conv2d(mid, 1 + 4 + num_classes, 1)

    def forward(self, f):
        x = self.tower(f)
        return self.pred(x)


# ─── Full model ───

class PerceptionModelV2(nn.Module):
    def __init__(self,
                 num_seg_classes: int = NUM_SEG_CLASSES,
                 num_det_classes: int = NUM_DET_CLASSES,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = MobileBackbone(pretrained=pretrained)
        self.seg_head = SegHead(
            self.backbone.ch_s8, self.backbone.ch_s16, self.backbone.ch_s32,
            num_classes=num_seg_classes,
        )
        # Det heads — one per stride, but from different backbone scales
        self.det_head_s8  = DetHead(self.backbone.ch_s8,  num_classes=num_det_classes)
        self.det_head_s16 = DetHead(self.backbone.ch_s16, num_classes=num_det_classes)
        self.det_head_s32 = DetHead(self.backbone.ch_s32, num_classes=num_det_classes)
        self.num_det_classes = num_det_classes

    def forward(self, x):
        _, _, H, W = x.shape
        f8, f16, f32 = self.backbone(x)
        seg_logits = self.seg_head(f8, f16, f32, out_size=(H, W))
        det_preds = [
            self.det_head_s8(f8),
            self.det_head_s16(f16),
            self.det_head_s32(f32),
        ]
        return {"seg": seg_logits, "det": det_preds}


# ─── Loss functions ───

def seg_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over seg classes. targets are long, 255 = ignore."""
    return F.cross_entropy(logits, targets, ignore_index=255)


def det_loss_fn(preds: list, targets: list,
                num_classes: int = NUM_DET_CLASSES) -> dict:
    """FCOS-style loss: BCE for objectness, L1 for box regression (l,t,r,b),
    cross-entropy for class. targets: list of length B, each dict with
    {boxes: (N, 4) in pixel coords, labels: (N,)}."""
    device = preds[0].device
    total_obj = torch.tensor(0.0, device=device)
    total_reg = torch.tensor(0.0, device=device)
    total_cls = torch.tensor(0.0, device=device)
    pos_count = 0

    for pred, stride in zip(preds, STRIDES):
        B, C, Hp, Wp = pred.shape
        obj = pred[:, 0]
        reg = pred[:, 1:5]       # l, t, r, b
        cls = pred[:, 5:]        # (B, NC, Hp, Wp)

        # Targets — float32 to avoid AMP dtype mismatches
        obj_tgt = torch.zeros_like(obj, dtype=torch.float32)
        reg_tgt = torch.zeros_like(reg, dtype=torch.float32)
        cls_tgt = torch.full((B, Hp, Wp), -1, dtype=torch.long, device=device)

        # Grid in pixel coords
        ys = (torch.arange(Hp, device=device).float() + 0.5) * stride
        xs = (torch.arange(Wp, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        for b in range(B):
            tgt = targets[b]
            boxes = tgt.get("boxes")
            labels = tgt.get("labels")
            if boxes is None or len(boxes) == 0:
                continue
            for bi in range(len(boxes)):
                x1, y1, x2, y2 = boxes[bi]
                inside = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
                if inside.sum() == 0:
                    continue
                obj_tgt[b][inside] = 1.0
                reg_tgt[b, 0][inside] = (grid_x[inside] - x1) / stride
                reg_tgt[b, 1][inside] = (grid_y[inside] - y1) / stride
                reg_tgt[b, 2][inside] = (x2 - grid_x[inside]) / stride
                reg_tgt[b, 3][inside] = (y2 - grid_y[inside]) / stride
                cls_tgt[b][inside] = int(labels[bi])

        # Cast preds to float32 for loss (AMP safety)
        obj_f = obj.float()
        reg_f = reg.float()
        cls_f = cls.float()

        total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_f, obj_tgt)

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

    n = len(preds)
    return {
        "obj": total_obj / n,
        "reg": total_reg / max(n, 1),
        "cls": total_cls / max(n, 1),
        "total": total_obj / n + total_reg / max(n, 1) + total_cls / max(n, 1),
        "pos_count": pos_count,
    }


# ─── Decoders for inference ───

@torch.no_grad()
def decode_segmentation(seg_logits: torch.Tensor) -> torch.Tensor:
    """Returns class indices (B, H, W)."""
    return seg_logits.argmax(dim=1)


@torch.no_grad()
def decode_detection(preds: list, conf_thr: float = 0.3,
                     iou_thr: float = 0.5) -> list:
    """Returns list of length B, each a list of dicts {x1,y1,x2,y2,score,cls}
    in pixel coords of the model input size."""
    from torchvision.ops import nms
    B = preds[0].shape[0]
    out = [[] for _ in range(B)]
    device = preds[0].device

    for pred, stride in zip(preds, STRIDES):
        _, C, Hp, Wp = pred.shape
        obj = torch.sigmoid(pred[:, 0])
        reg = pred[:, 1:5]
        cls_logits = pred[:, 5:]
        cls_probs = torch.softmax(cls_logits, dim=1)
        cls_scores, cls_ids = cls_probs.max(dim=1)

        score = obj * cls_scores  # (B, Hp, Wp)
        mask = score > conf_thr

        ys = (torch.arange(Hp, device=device).float() + 0.5) * stride
        xs = (torch.arange(Wp, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        for b in range(B):
            m = mask[b]
            if m.sum() == 0:
                continue
            cy = grid_y[m]
            cx = grid_x[m]
            l = reg[b, 0][m] * stride
            t = reg[b, 1][m] * stride
            r = reg[b, 2][m] * stride
            bt = reg[b, 3][m] * stride
            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + bt
            s = score[b][m]
            c = cls_ids[b][m]

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            keep = nms(boxes, s, iou_thr)
            for k in keep.tolist():
                out[b].append({
                    "x1": float(x1[k]), "y1": float(y1[k]),
                    "x2": float(x2[k]), "y2": float(y2[k]),
                    "score": float(s[k]),
                    "cls": int(c[k]),
                })

    return out


def build_model(pretrained: bool = True) -> PerceptionModelV2:
    return PerceptionModelV2(pretrained=pretrained)
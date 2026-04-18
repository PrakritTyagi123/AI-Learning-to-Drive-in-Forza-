"""
ForzaTek AI — Perception Model Training
=========================================
Standalone training script. No UI, no subprocess management — just:

    python train_perception.py

It trains the small perception model on your labeled frames, saves the best
checkpoint to models/perception_v1.pt, and prints per-epoch progress.

Options:
    --epochs N           number of epochs (default 60)
    --batch-size N       batch size (default 16)
    --lr X               learning rate (default 3e-4)
    --resume PATH        resume from a checkpoint
    --workers N          dataloader workers (default 4)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Let the script run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.perception import (
    PerceptionModelV2, seg_loss_fn, det_loss_fn,
    decode_segmentation, decode_detection, INPUT_W, INPUT_H,
    NUM_SEG_CLASSES, NUM_DET_CLASSES,
)
from backend.perception_dataset import PerceptionDataset, collate_fn, make_splits


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[float]:
    """Per-class IoU, averaged over batch."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        tgt_c = (target == c)
        inter = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


def train(epochs: int, batch_size: int, lr: float, num_workers: int,
          resume: str | None, progress_file: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Device: {device}")

    def _write_progress(state: dict):
        if not progress_file:
            return
        try:
            p = Path(progress_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state), encoding="utf-8")
        except Exception:
            pass

    _start_time = time.time()
    _write_progress({
        "running": True, "current_epoch": 0, "total_epochs": epochs,
        "train_loss": 0.0, "val_miou": 0.0, "val_road_iou": 0.0,
        "val_det_conf": 0.0, "best_road_iou": 0.0, "started_at": _start_time,
        "status": "initializing", "history": [],
    })

    train_ids, val_ids = make_splits(val_ratio=0.15)
    print(f"[TRAIN] {len(train_ids)} train frames / {len(val_ids)} val frames")

    train_ds = PerceptionDataset(train_ids, training=True)
    val_ds   = PerceptionDataset(val_ids,   training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    model = PerceptionModelV2(pretrained=True).to(device)
    if resume and Path(resume).exists():
        print(f"[TRAIN] Resuming from {resume}")
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_miou = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # ─── TRAIN ───
        model.train()
        run_loss = 0.0
        run_seg = 0.0
        run_obj = 0.0
        run_reg = 0.0
        run_cls = 0.0
        n_batches = 0

        for imgs, segs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            segs = segs.to(device, non_blocking=True)
            targets_dev = []
            for t in targets:
                targets_dev.append({
                    "boxes":  t["boxes"].to(device),
                    "labels": t["labels"].to(device),
                })

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs)
                ls = seg_loss_fn(out["seg"], segs)
                ld = det_loss_fn(out["det"], targets_dev)
                loss = ls + ld["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            run_loss += float(loss.item())
            run_seg  += float(ls.item())
            run_obj  += float(ld["obj"].item())
            run_reg  += float(ld["reg"].item())
            run_cls  += float(ld["cls"].item())
            n_batches += 1

        scheduler.step()

        # ─── VAL ───
        model.eval()
        all_ious = []
        det_confs = []
        with torch.no_grad():
            for imgs, segs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                segs = segs.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(imgs)
                pred_seg = decode_segmentation(out["seg"])
                all_ious.append(compute_iou(pred_seg, segs, NUM_SEG_CLASSES))

                dets = decode_detection(out["det"], conf_thr=0.3)
                for per_img in dets:
                    if per_img:
                        avg = sum(d["score"] for d in per_img) / len(per_img)
                        det_confs.append(avg)

        iou_arr = np.array(all_ious, dtype=float)
        per_class_miou = np.nanmean(iou_arr, axis=0)
        mean_miou = float(np.nanmean(per_class_miou))
        mean_det_conf = float(np.mean(det_confs)) if det_confs else 0.0

        dt = time.time() - t0
        tl = run_loss / max(n_batches, 1)
        print(
            f"[TRAIN] ep {epoch:3d}/{epochs} | "
            f"loss={tl:.4f} (seg={run_seg/max(n_batches,1):.3f} "
            f"obj={run_obj/max(n_batches,1):.3f} "
            f"reg={run_reg/max(n_batches,1):.3f} "
            f"cls={run_cls/max(n_batches,1):.3f}) | "
            f"val mIoU={mean_miou:.3f} "
            f"(offroad={per_class_miou[0]:.3f} road={per_class_miou[1]:.3f}) | "
            f"det_conf={mean_det_conf:.2f} | {dt:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "loss": tl,
            "val_miou": mean_miou,
            "val_miou_road": float(per_class_miou[1]),
            "val_miou_offroad": float(per_class_miou[0]),
            "mean_det_conf": mean_det_conf,
        })

        # Write live progress for the UI
        elapsed_s = time.time() - _start_time
        per_epoch = elapsed_s / max(epoch, 1)
        eta_s = (epochs - epoch) * per_epoch
        _write_progress({
            "running": True,
            "current_epoch": epoch,
            "total_epochs": epochs,
            "train_loss": tl,
            "val_miou": mean_miou,
            "val_road_iou": float(per_class_miou[1]),
            "val_offroad_iou": float(per_class_miou[0]),
            "val_det_conf": mean_det_conf,
            "best_road_iou": best_miou if per_class_miou[1] <= best_miou else float(per_class_miou[1]),
            "started_at": _start_time,
            "eta_sec": eta_s,
            "status": "training",
            "history": history,
        })

        # Save best by road-class IoU (that's what matters for driving)
        if per_class_miou[1] > best_miou:
            best_miou = float(per_class_miou[1])
            ckpt_path = MODELS_DIR / "perception_v1.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_miou_road": best_miou,
                "history": history,
            }, ckpt_path)
            print(f"[TRAIN] -> saved new best to {ckpt_path} (road IoU {best_miou:.3f})")

        # Also save a "last" checkpoint every epoch
        last_path = MODELS_DIR / "perception_v1_last.pt"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "history": history,
        }, last_path)

    # Save history
    (MODELS_DIR / "perception_v1_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    _write_progress({
        "running": False,
        "current_epoch": epochs,
        "total_epochs": epochs,
        "train_loss": history[-1]["loss"] if history else 0.0,
        "val_miou": history[-1]["val_miou"] if history else 0.0,
        "val_road_iou": history[-1]["val_miou_road"] if history else 0.0,
        "val_offroad_iou": history[-1]["val_miou_offroad"] if history else 0.0,
        "val_det_conf": history[-1]["mean_det_conf"] if history else 0.0,
        "best_road_iou": best_miou,
        "started_at": _start_time,
        "eta_sec": 0.0,
        "status": "done",
        "history": history,
    })
    print(f"[TRAIN] Done. Best road IoU: {best_miou:.3f}")
    print(f"[TRAIN] Best ckpt: {MODELS_DIR / 'perception_v1.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--progress-file", type=str, default=None,
                   help="Path to JSON file for writing live progress (for UI)")
    args = p.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        resume=args.resume,
        progress_file=args.progress_file,
    )
"""
ForzaTek AI — Training Script
==============================
Trains (or fine-tunes) the multi-task perception model on labeled frames.

Usage:
  python -m training.train                                 # train from scratch
  python -m training.train --resume models/round_1.pt      # fine-tune
  python -m training.train --round 2 --epochs 30
  python -m training.train --progress-file training/_progress.json

Pipeline:
  1. Query DB for frames with both seg and det labels    (dataset.get_frame_splits)
  2. Build train/val split stratified by game_version
  3. Decode JPEGs + labels on the fly                     (dataset.LabeledFramesDataset)
  4. Train with AdamW + cosine LR schedule, mixed precision
  5. Save the best checkpoint by val mIoU
  6. Register the new model in DB (and optionally set active)

Progress file (written each epoch for the UI's live dashboard):
  {
    "running":       true,
    "current_epoch": 12,
    "total_epochs":  30,
    "train_loss":    0.247,
    "val_miou":      0.791,
    "val_conf":      0.68,
  }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import DB_PATH, init_db, write_conn
from training.model import (
    PerceptionModel, seg_loss, det_loss,
    NUM_SEG_CLASSES,
    decode_segmentation, decode_detection,
)
from training.dataset import LabeledFramesDataset, collate, get_frame_splits
from training.metrics import compute_iou


def _write_progress(path: Path, **fields) -> None:
    """Write the progress JSON atomically so the UI never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(fields), encoding="utf-8")
    tmp.replace(path)


def train(
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 3e-4,
    resume: str | None = None,
    round_num: int = 1,
    save_name: str | None = None,
    set_active: bool = True,
    progress_file: str | None = None,
) -> tuple[int, float]:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    prog_path = Path(progress_file) if progress_file else None

    init_db(DB_PATH)
    train_ids, val_ids = get_frame_splits()
    if len(train_ids) < 10:
        raise RuntimeError(
            f"Not enough labeled frames to train. Have {len(train_ids)} train + "
            f"{len(val_ids)} val. Need at least ~50 for meaningful training."
        )
    print(f"Training on {len(train_ids)} frames, validating on {len(val_ids)}.")

    train_ds = LabeledFramesDataset(train_ids, augment=True)
    val_ds   = LabeledFramesDataset(val_ids,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = PerceptionModel().to(device)
    if resume:
        print(f"Resuming from {resume}")
        sd = torch.load(resume, map_location=device)
        model.load_state_dict(sd)

    optim = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_score = -1.0
    best_conf  = 0.0
    save_name = save_name or f"round_{round_num}.pt"
    Path("models").mkdir(exist_ok=True)
    save_path = Path("models") / save_name

    history: list[dict] = []

    if prog_path:
        _write_progress(
            prog_path, running=True,
            current_epoch=0, total_epochs=epochs,
            train_loss=0.0, val_miou=0.0, val_conf=0.0,
        )

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for imgs, segs, dets in train_loader:
            imgs = imgs.to(device)
            segs = segs.to(device)
            dets_dev = [{"boxes": d["boxes"].to(device), "labels": d["labels"].to(device)}
                        for d in dets]

            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(imgs)
                ls = seg_loss(out["seg"], segs)
                ld = det_loss(out["det"], dets_dev)
                loss = 1.0 * ls + 1.0 * ld

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss /= max(1, n)
        sched.step()

        # ─── Validation: mIoU + mean detection confidence ───
        model.eval()
        ious: list[float] = []
        det_confs: list[float] = []
        with torch.no_grad():
            for imgs, segs, _ in val_loader:
                imgs = imgs.to(device)
                segs = segs.to(device)
                out = model(imgs)
                pred = decode_segmentation(out["seg"])
                ious.extend(compute_iou(pred, segs, NUM_SEG_CLASSES))

                # Detection confidence: average max-score per image
                dets = decode_detection(out["det"], conf_threshold=0.30)
                for d in dets:
                    scores = d.get("scores")
                    if scores is not None and len(scores) > 0:
                        det_confs.append(float(scores.mean().item()))

        ious_clean = [i for i in ious if not np.isnan(i)]
        mean_iou = float(np.mean(ious_clean)) if ious_clean else 0.0
        mean_det_conf = float(np.mean(det_confs)) if det_confs else 0.0

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{epochs} | loss={train_loss:.4f} | "
              f"val mIoU={mean_iou:.3f} | det conf={mean_det_conf:.2f} | {elapsed:.1f}s")
        history.append({
            "epoch":        epoch + 1,
            "loss":         train_loss,
            "mIoU":         mean_iou,
            "mean_det_conf": mean_det_conf,
        })

        if prog_path:
            _write_progress(
                prog_path, running=True,
                current_epoch=epoch + 1, total_epochs=epochs,
                train_loss=train_loss, val_miou=mean_iou, val_conf=mean_det_conf,
            )

        if mean_iou > best_score:
            best_score = mean_iou
            best_conf  = mean_det_conf
            import torch as _torch
            _torch.save(model.state_dict(), save_path)
            print(f"  ✓ saved new best to {save_path}")

    # ─── Register model in DB ───
    metrics = {
        "best_mIoU":     best_score,
        "mean_det_conf": best_conf,
        "history":       history,
    }
    with write_conn(DB_PATH) as c:
        cur = c.execute(
            """INSERT INTO models
               (name, round_num, path, trained_on, metrics_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (save_name, round_num, str(save_path), len(train_ids),
             json.dumps(metrics), time.time())
        )
        model_id = cur.lastrowid
        if set_active:
            c.execute("UPDATE models SET is_active=0")
            c.execute("UPDATE models SET is_active=1 WHERE id=?", (model_id,))

    if prog_path:
        _write_progress(
            prog_path, running=False,
            current_epoch=epochs, total_epochs=epochs,
            train_loss=train_loss, val_miou=best_score, val_conf=best_conf,
        )

    print(f"\nTraining complete. Best mIoU: {best_score:.3f} (det conf {best_conf:.2f})")
    print(f"Model registered with id={model_id}")
    return model_id, best_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--progress-file", type=str, default=None)
    args = ap.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        round_num=args.round,
        save_name=args.name,
        progress_file=args.progress_file,
    )

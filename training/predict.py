"""
ForzaTek AI — Predict & Active Learning
========================================
After training a model, run it on unlabeled frames:
  1. Write proposals to DB
  2. Flip frame status to 'proposed'
  3. Queue the top-K most uncertain into active_queue for user review

Usage:
  python -m training.predict --model models/round_1.pt --limit 500
  python -m training.predict --use-active       # uses current active model from DB
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import DB_PATH, init_db, read_conn, write_conn, get_active_model
from training.model import (
    PerceptionModel, decode_segmentation, decode_detection,
    frame_uncertainty, INPUT_H, INPUT_W,
)


def preprocess(frame_bgr: np.ndarray):
    import torch
    img = cv2.resize(frame_bgr, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def encode_seg_mask(mask: np.ndarray, orig_h: int, orig_w: int) -> str:
    """Resize to original dims, PNG-encode, base64."""
    resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", resized)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def detections_to_payload(dets: list[dict], orig_w: int, orig_h: int) -> dict:
    """Convert pixel-space detections at model input size to normalized boxes at orig size."""
    sx = orig_w / INPUT_W
    sy = orig_h / INPUT_H
    boxes = []
    cls_names = ["vehicle", "sign"]
    for d in dets:
        x1 = d["x1"] * sx; x2 = d["x2"] * sx
        y1 = d["y1"] * sy; y2 = d["y2"] * sy
        boxes.append({
            "cls": cls_names[d["cls"]] if d["cls"] < len(cls_names) else "unknown",
            "x": float(x1 / orig_w),
            "y": float(y1 / orig_h),
            "w": float((x2 - x1) / orig_w),
            "h": float((y2 - y1) / orig_h),
            "confidence": float(d["score"]),
        })
    return {"boxes": boxes}


def run_predictions(model_path: str, limit: int = 500, queue_top_k: int = 50,
                    round_num: int = 1, model_id: int | None = None):
    import torch

    init_db(DB_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PerceptionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Pick unlabeled frames
    with read_conn(DB_PATH) as c:
        rows = c.execute(
            """SELECT id, frame_jpeg, width, height FROM frames
               WHERE label_status='unlabeled'
               ORDER BY RANDOM() LIMIT ?""",
            (limit,),
        ).fetchall()

    print(f"Predicting on {len(rows)} frames...")
    now = time.time()
    frame_uncertainties = []

    with torch.no_grad():
        for r in rows:
            arr = np.frombuffer(bytes(r["frame_jpeg"]), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            x = preprocess(frame).to(device)
            out = model(x)
            seg_logits = out["seg"]
            det_preds = out["det"]

            pred_seg = decode_segmentation(seg_logits)[0].cpu().numpy()
            pred_det = decode_detection(det_preds)[0]

            unc = frame_uncertainty(seg_logits, det_preds)
            frame_uncertainties.append((r["id"], unc))

            seg_payload = {
                "mask_png_b64": encode_seg_mask(pred_seg, r["height"], r["width"]),
                "classes": ["offroad", "road", "curb", "wall"],
            }
            det_payload = detections_to_payload(pred_det, r["width"], r["height"])

            with write_conn(DB_PATH) as c:
                c.execute("DELETE FROM proposals WHERE frame_id=? AND task IN ('seg','det')",
                          (r["id"],))
                c.execute(
                    """INSERT INTO proposals
                       (frame_id, task, data_json, confidence, uncertainty, model_id, created_at)
                       VALUES (?, 'seg', ?, ?, ?, ?, ?)""",
                    (r["id"], json.dumps(seg_payload), 1.0 - unc, unc, model_id, now)
                )
                c.execute(
                    """INSERT INTO proposals
                       (frame_id, task, data_json, confidence, uncertainty, model_id, created_at)
                       VALUES (?, 'det', ?, ?, ?, ?, ?)""",
                    (r["id"], json.dumps(det_payload), 1.0 - unc, unc, model_id, now)
                )
                c.execute("UPDATE frames SET label_status='proposed' WHERE id=?", (r["id"],))

    # Pick top-K most uncertain for active learning queue
    frame_uncertainties.sort(key=lambda x: -x[1])
    top_k = frame_uncertainties[:queue_top_k]

    with write_conn(DB_PATH) as c:
        c.execute("DELETE FROM active_queue")
        for fid, unc in top_k:
            c.execute(
                """INSERT OR REPLACE INTO active_queue
                   (frame_id, uncertainty, queued_at, round_num)
                   VALUES (?, ?, ?, ?)""",
                (fid, unc, now, round_num)
            )

    print(f"Wrote {len(rows)} proposals.")
    print(f"Queued top {len(top_k)} most uncertain for review.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--use-active", action="store_true")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--queue-top-k", type=int, default=50)
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    if args.use_active:
        active = get_active_model()
        if not active:
            print("No active model. Train one first.")
            sys.exit(1)
        path = active["path"]
        mid = active["id"]
    else:
        if not args.model:
            print("Pass --model PATH or --use-active")
            sys.exit(1)
        path = args.model
        mid = None

    run_predictions(path, limit=args.limit, queue_top_k=args.queue_top_k,
                    round_num=args.round, model_id=mid)

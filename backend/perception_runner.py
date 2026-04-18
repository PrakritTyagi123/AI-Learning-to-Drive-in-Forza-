"""
ForzaTek AI — Perception Runner
================================
Backend for the Train + Test UI pages.

Routes:
  POST /api/perception/train          — kick off a training run
  POST /api/perception/cancel         — terminate the running job
  GET  /api/perception/progress       — live progress (epoch, losses, IoU, ETA)
  GET  /api/perception/log            — last 200 lines of training log
  GET  /api/perception/stats          — labeled-frame counts & readiness
  GET  /api/perception/test_frame     — fetch a frame with predictions + labels
                                          overlayed, for visual inspection
"""
from __future__ import annotations

import base64
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from backend.database import DB_PATH, read_conn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = PROJECT_ROOT / "models" / "_perception_progress.json"
LOG_FILE = PROJECT_ROOT / "models" / "_perception_log.txt"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


_state = {
    "proc": None,
    "started_at": 0.0,
    "log_file_handle": None,
    "last_exit_code": None,
}
_lock = threading.Lock()


def _running() -> bool:
    p = _state["proc"]
    return p is not None and p.poll() is None


def _read_progress() -> Dict[str, Any]:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _spawn(params: Dict[str, Any]) -> subprocess.Popen:
    """Launch `python train_perception.py` with given params."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Reset progress
    PROGRESS_FILE.write_text(json.dumps({
        "running": True, "current_epoch": 0,
        "total_epochs": int(params.get("epochs", 60)),
        "train_loss": 0.0, "val_miou": 0.0,
        "val_road_iou": 0.0, "val_det_conf": 0.0,
        "best_road_iou": 0.0, "status": "starting",
        "history": [],
    }), encoding="utf-8")

    cmd = [
        sys.executable, "train_perception.py",
        "--epochs", str(int(params.get("epochs", 60))),
        "--batch-size", str(int(params.get("batch_size", 16))),
        "--lr", str(params.get("lr", 3e-4)),
        "--workers", str(int(params.get("workers", 4))),
        "--progress-file", str(PROGRESS_FILE),
    ]
    if params.get("resume"):
        cmd += ["--resume", str(params["resume"])]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    log_file = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
    log_file.write(f"Launching: {' '.join(cmd)}\n\n")
    log_file.flush()

    print(f"[PERCEPTION] Starting training subprocess. Log: {LOG_FILE}")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    _state["log_file_handle"] = log_file
    return proc


# ─── Test frame visualization ───

def _load_frame_with_labels(frame_id: int):
    """Load a frame, its labeled seg mask, and detection boxes from the DB."""
    with read_conn(DB_PATH) as c:
        row = c.execute(
            "SELECT frame_jpeg, width, height, label_status FROM frames WHERE id=?",
            (frame_id,),
        ).fetchone()
        if not row:
            return None
        labels = c.execute(
            "SELECT task, data_json FROM labels WHERE frame_id=?",
            (frame_id,),
        ).fetchall()
        proposals = c.execute(
            "SELECT task, data_json FROM proposals WHERE frame_id=?",
            (frame_id,),
        ).fetchall()

    jpg = bytes(row["frame_jpeg"])
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Parse GT labels first, fall back to proposals
    gt_seg_mask = None
    gt_boxes = []
    for l in list(labels) + list(proposals):
        task = l["task"]
        data = json.loads(l["data_json"])
        if task == "seg" and gt_seg_mask is None:
            mb64 = data.get("mask_png_b64", "")
            if mb64:
                m = base64.b64decode(mb64)
                ma = np.frombuffer(m, dtype=np.uint8)
                mask = cv2.imdecode(ma, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                    gt_seg_mask = (mask == 1).astype(np.uint8)
        elif task == "det" and not gt_boxes:
            for b in data.get("boxes", []):
                gt_boxes.append({
                    "cls": b.get("cls", "vehicle"),
                    "x": float(b.get("x", 0)),
                    "y": float(b.get("y", 0)),
                    "w": float(b.get("w", 0)),
                    "h": float(b.get("h", 0)),
                })

    return {
        "img_bgr": img_bgr,
        "gt_seg_mask": gt_seg_mask,
        "gt_boxes": gt_boxes,
        "width": row["width"],
        "height": row["height"],
        "status": row["label_status"],
    }


def _render_overlay(img_bgr, road_mask, boxes, alpha: float = 0.5) -> np.ndarray:
    """Overlay road mask (green) and boxes on a BGR image. Returns BGR."""
    out = img_bgr.copy()
    H, W = out.shape[:2]

    if road_mask is not None:
        # Make sure mask is the same size as the image
        if road_mask.shape[:2] != (H, W):
            road_mask = cv2.resize(road_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = out.copy()
        overlay[road_mask > 0] = (0, 200, 80)  # green
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    for b in boxes or []:
        # boxes may be either {x,y,w,h} normalized or {x1,y1,x2,y2} pixel
        if "x1" in b:
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        else:
            x1 = int(b["x"] * W)
            y1 = int(b["y"] * H)
            x2 = int((b["x"] + b["w"]) * W)
            y2 = int((b["y"] + b["h"]) * H)
        cls_name = b.get("cls_name") or b.get("cls", "?")
        if isinstance(cls_name, int):
            cls_name = ["vehicle", "sign"][cls_name] if cls_name < 2 else str(cls_name)
        color = (0, 180, 255) if cls_name == "vehicle" else (0, 120, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = cls_name
        if "score" in b:
            label += f" {b['score']:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def _jpeg_b64(img_bgr: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if pred_mask is None or gt_mask is None:
        return 0.0
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return inter / union if union > 0 else 0.0


# Cached runtime (lazy-loaded once, reused across calls)
_runtime = None
_runtime_path = None


def _get_runtime():
    """Lazy-load PerceptionRuntime from the active checkpoint."""
    global _runtime, _runtime_path
    ckpt = MODELS_DIR / "perception_v1.pt"
    if not ckpt.exists():
        return None
    # Reload if file changed
    if _runtime is None or _runtime_path != str(ckpt) or _runtime_path_mtime() != ckpt.stat().st_mtime:
        try:
            from backend.perception_infer import PerceptionRuntime
            _runtime = PerceptionRuntime(str(ckpt))
            _runtime_path = str(ckpt)
            _runtime._loaded_mtime = ckpt.stat().st_mtime
        except Exception as e:
            print(f"[PERCEPTION] Failed to load runtime: {e}")
            _runtime = None
    return _runtime


def _runtime_path_mtime():
    if _runtime is None:
        return None
    return getattr(_runtime, "_loaded_mtime", None)


# ─── Routes ───

def register_routes(app: FastAPI) -> None:

    @app.get("/api/perception/stats")
    def stats():
        with read_conn(DB_PATH) as c:
            total = c.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            labeled = c.execute(
                "SELECT COUNT(*) FROM frames WHERE label_status='labeled'"
            ).fetchone()[0]
            seg_count = c.execute("SELECT COUNT(*) FROM labels WHERE task='seg'").fetchone()[0]
            det_count = c.execute("SELECT COUNT(*) FROM labels WHERE task='det'").fetchone()[0]
            both_count = c.execute(
                "SELECT COUNT(DISTINCT l1.frame_id) FROM labels l1 "
                "JOIN labels l2 ON l2.frame_id=l1.frame_id "
                "WHERE l1.task='seg' AND l2.task='det'"
            ).fetchone()[0]

        ckpt_exists = (MODELS_DIR / "perception_v1.pt").exists()
        ckpt_info = None
        if ckpt_exists:
            try:
                import torch as _torch
                state = _torch.load(MODELS_DIR / "perception_v1.pt", map_location="cpu")
                ckpt_info = {
                    "path": str(MODELS_DIR / "perception_v1.pt"),
                    "epoch": state.get("epoch"),
                    "best_miou_road": state.get("best_miou_road"),
                    "size_mb": round(
                        (MODELS_DIR / "perception_v1.pt").stat().st_size / (1024 * 1024), 1
                    ),
                }
            except Exception:
                pass

        return {
            "total_frames": total,
            "labeled_frames": labeled,
            "seg_labels": seg_count,
            "det_labels": det_count,
            "both_labels": both_count,
            "ready_to_train": both_count >= 100,
            "checkpoint": ckpt_info,
        }

    @app.post("/api/perception/train")
    def start(payload: dict = None):
        payload = payload or {}
        with _lock:
            if _running():
                raise HTTPException(409, "Training already in progress.")
            try:
                proc = _spawn(payload)
            except FileNotFoundError as e:
                raise HTTPException(500, f"Could not launch training: {e}")
            _state["proc"] = proc
            _state["started_at"] = time.time()
        return {"ok": True, "pid": proc.pid}

    @app.post("/api/perception/cancel")
    def cancel():
        with _lock:
            p = _state["proc"]
            if p is None or p.poll() is not None:
                return {"ok": True, "already_stopped": True}
            try:
                if os.name == "nt":
                    p.terminate()
                else:
                    p.terminate()
            except Exception as e:
                raise HTTPException(500, f"Could not terminate: {e}")
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        return {"ok": True}

    @app.get("/api/perception/progress")
    def progress():
        prog = _read_progress()
        running = _running()
        if not running and _state["proc"] is not None:
            _state["last_exit_code"] = _state["proc"].returncode
            _state["proc"] = None
            # Close log file
            h = _state.get("log_file_handle")
            if h:
                try: h.close()
                except Exception: pass
                _state["log_file_handle"] = None

        # If process crashed, mark as not running in the progress payload
        if not running:
            prog = dict(prog)
            prog["running"] = False
            if _state.get("last_exit_code") not in (None, 0):
                prog["status"] = "crashed"
                prog["exit_code"] = _state["last_exit_code"]

        return prog

    @app.get("/api/perception/log")
    def get_log():
        if not LOG_FILE.exists():
            return {"log": "(no log yet)"}
        try:
            lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            return {"log": "\n".join(lines[-300:])}
        except Exception as e:
            return {"log": f"(error reading log: {e})"}

    @app.get("/api/perception/test_frame")
    def test_frame(id: int = None):
        """Return a frame rendered with GT (from labels) and model prediction overlays.
        If id is None, picks a random labeled frame."""
        with read_conn(DB_PATH) as c:
            if id is None:
                r = c.execute(
                    "SELECT id FROM frames WHERE label_status='labeled' "
                    "ORDER BY RANDOM() LIMIT 1"
                ).fetchone()
                if not r:
                    raise HTTPException(404, "No labeled frames available")
                id = r["id"]

        data = _load_frame_with_labels(id)
        if data is None:
            raise HTTPException(404, f"Frame {id} not found")

        img_bgr = data["img_bgr"]
        gt_mask = data["gt_seg_mask"]
        gt_boxes = data["gt_boxes"]

        gt_overlay = _render_overlay(img_bgr, gt_mask, gt_boxes)

        # Run model prediction if checkpoint exists
        pred_b64 = None
        pred_iou = None
        pred_det_count = 0
        rt = _get_runtime()
        if rt is not None:
            try:
                result = rt.infer(img_bgr)
                pred_mask = result["road_mask"]
                pred_boxes = result["boxes"]
                pred_overlay = _render_overlay(img_bgr, pred_mask, pred_boxes)
                pred_b64 = _jpeg_b64(pred_overlay)
                pred_iou = _compute_iou(pred_mask, gt_mask) if gt_mask is not None else None
                pred_det_count = len(pred_boxes)
            except Exception as e:
                print(f"[PERCEPTION] Inference failed: {e}")

        return {
            "frame_id": id,
            "status": data["status"],
            "orig_b64": _jpeg_b64(img_bgr),
            "gt_overlay_b64": _jpeg_b64(gt_overlay),
            "pred_overlay_b64": pred_b64,
            "pred_iou": pred_iou,
            "pred_det_count": pred_det_count,
            "gt_det_count": len(gt_boxes),
            "has_model": rt is not None,
        }
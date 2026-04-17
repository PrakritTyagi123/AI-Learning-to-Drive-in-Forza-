"""
ForzaTek AI — Training Runner
==============================
HTTP-accessible wrapper around `training/train.py`.
Spawns training as a child Python process so the UI can start / stop /
monitor runs without blocking the FastAPI event loop.

Progress is tracked in-memory (single active job at a time). The child
process writes a JSON status line to training/_progress.json every few
seconds; we read that file to answer /api/train/progress.

Routes:
  POST /api/train/start    — kick off a training round
  POST /api/train/cancel   — terminate the running job
  GET  /api/train/progress — current epoch, loss, mIoU, ETA
  GET  /api/train/models   — list all checkpoints (from the `models` table)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from backend.database import read_conn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = PROJECT_ROOT / "training" / "_progress.json"
MODELS_DIR = PROJECT_ROOT / "models"


_state = {
    "proc": None,            # Popen or None
    "started_at": 0.0,
    "params": {},
    "cancelled": False,
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


def _estimate_eta(prog: Dict[str, Any]) -> float:
    cur = prog.get("current_epoch") or 0
    total = prog.get("total_epochs") or 1
    started = _state["started_at"]
    if cur <= 0 or not started:
        return 0.0
    elapsed = time.time() - started
    per_epoch = elapsed / max(cur, 1)
    return max(0.0, (total - cur) * per_epoch)


def _spawn(params: Dict[str, Any]) -> subprocess.Popen:
    """Launch `python -m training.train` with the given params."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Reset progress file so a stale previous run doesn't leak through
    PROGRESS_FILE.write_text(json.dumps({
        "running": True,
        "current_epoch": 0,
        "total_epochs": int(params.get("epochs", 30)),
        "train_loss": 0.0,
        "val_miou": 0.0,
        "val_conf": 0.0,
    }), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "training.train",
        "--round", str(int(params.get("round_num", 1))),
        "--epochs", str(int(params.get("epochs", 30))),
        "--batch-size", str(int(params.get("batch_size", 8))),
        "--lr", str(params.get("lr", 3e-4)),
        "--progress-file", str(PROGRESS_FILE),
    ]
    if params.get("resume"):
        cmd += ["--resume", str(params["resume"])]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


# ─── Routes ───────────────────────────────────────────────────────────────

def register_routes(app: FastAPI) -> None:
    @app.post("/api/train/start")
    async def start(payload: dict):
        with _lock:
            if _running():
                raise HTTPException(409, "A training run is already in progress.")
            params = dict(payload or {})
            try:
                proc = _spawn(params)
            except FileNotFoundError as e:
                raise HTTPException(500, f"Could not launch training: {e}")
            _state["proc"] = proc
            _state["started_at"] = time.time()
            _state["params"] = params
            _state["cancelled"] = False
        return {"ok": True, "pid": proc.pid}

    @app.post("/api/train/cancel")
    def cancel():
        with _lock:
            p = _state["proc"]
            if p is None or p.poll() is not None:
                return {"ok": True, "already_stopped": True}
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.terminate()
                _state["cancelled"] = True
            except Exception as e:
                raise HTTPException(500, f"Could not terminate: {e}")
        # Give it a moment, then hard-kill
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        return {"ok": True}

    @app.get("/api/train/progress")
    def progress():
        prog = _read_progress()
        running = _running()
        if not running and _state["proc"] is not None:
            _state["last_exit_code"] = _state["proc"].returncode
            _state["proc"] = None
        out = {
            "running": running,
            "current_epoch": int(prog.get("current_epoch", 0) or 0),
            "total_epochs":  int(prog.get("total_epochs", 0) or 0),
            "train_loss":    float(prog.get("train_loss", 0.0) or 0.0),
            "val_miou":      float(prog.get("val_miou", 0.0) or 0.0),
            "val_conf":      float(prog.get("val_conf", 0.0) or 0.0),
            "eta_sec":       _estimate_eta(prog) if running else 0.0,
            "last_exit_code": _state["last_exit_code"],
        }
        return out

    @app.get("/api/train/models")
    def list_models():
        with read_conn() as c:
            rows = c.execute(
                "SELECT id, name, path, round_num, is_active, created_at, "
                "       metrics_json, trained_on "
                "FROM models ORDER BY created_at DESC"
            ).fetchall()
        return {
            "models": [
                {
                    "id":          r["id"],
                    "name":        r["name"],
                    "path":        r["path"],
                    "round_num":   r["round_num"],
                    "is_active":   bool(r["is_active"]),
                    "created_at":  r["created_at"],
                    "metrics_json": r["metrics_json"] or "{}",
                    "trained_on":  r["trained_on"] or 0,
                }
                for r in rows
            ]
        }

    @app.post("/api/train/activate/{model_id}")
    def activate(model_id: int):
        from backend.database import write_conn
        with write_conn() as c:
            c.execute("UPDATE models SET is_active = 0")
            c.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
        return {"ok": True}

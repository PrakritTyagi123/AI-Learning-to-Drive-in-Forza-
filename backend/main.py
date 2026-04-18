"""
ForzaTek AI — Main App Entry Point
===================================
Single FastAPI server that exposes all backend functionality and serves
the frontend HTML files.

Routes are organized by subsystem:
  /api/record/*      — live Forza recording         (recorder.py)
  /api/ingest/*      — video / YouTube ingestion    (video_ingester.py)
  /api/label/*       — labeling tool backend        (labeling_backend.py)
  /api/telemetry/*   — Forza UDP telemetry          (telemetry_listener.py)
  /api/system/*      — stats, health, settings

Static pages:
  /           — dashboard
  /record     — recording panel
  /ingest     — ingest panel
  /label      — labeling tool
  /telemetry  — live telemetry dashboard
  /drive      — autonomous driving (placeholder)
  /settings   — app configuration
  /help       — in-app docs

Run with:
  python -m backend.main
  (or: uvicorn backend.main:app --host 0.0.0.0 --port 8000)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import init_db, overall_stats, DB_PATH
from backend import recorder, video_ingester, labeling_backend
from backend import telemetry_listener, perception_runner


app = FastAPI(title="ForzaTek AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init DB on import
init_db(DB_PATH)

# Register subsystem routes
recorder.register_routes(app)
video_ingester.register_routes(app)
labeling_backend.register_routes(app)
telemetry_listener.register_routes(app)
perception_runner.register_routes(app)

# Start the UDP telemetry listener in a daemon thread so /api/telemetry/live
# has data as soon as Forza sends a packet.
telemetry_listener.start_background_listener()


# ─── System routes ───

@app.get("/api/system/stats")
def system_stats():
    return overall_stats()


@app.get("/api/system/health")
def health():
    return {"ok": True, "version": "1.0"}


SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"
DEFAULT_SETTINGS = {
    "default_game_version": "fh5",
    "unit_system": "imperial",
    "backend_host": "http://localhost:8000",
    "autosave": "30",
    "db_path": "data/forzatek.db",
    "models_path": "models/",
    "videos_path": "data/videos/",
    "udp_port": "5300",
    "capture_backend": "mss",
    "capture_monitor": "1",
    "capture_fps": "60",
    "train_device": "cuda:0",
    "infer_device": "cuda:0",
    "precision": "fp16",
    "prelabel_thr": "0.35",
    "queue_size": "50",
    "det_conf_thr": "0.30",
}


@app.get("/api/system/settings")
def get_settings():
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_SETTINGS


@app.post("/api/system/settings")
async def save_settings(payload: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged = {**DEFAULT_SETTINGS, **(payload or {})}
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return {"ok": True}


# ─── Static frontend ───

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _page(name: str) -> FileResponse:
    p = FRONTEND_DIR / name
    if not p.exists():
        raise HTTPException(404, f"Frontend file missing: {name}")
    return FileResponse(p)


@app.get("/")
def index():
    return _page("dashboard.html")


@app.get("/record")
def record_page():
    return _page("record.html")


@app.get("/ingest")
def ingest_page():
    return _page("ingest.html")


@app.get("/label")
def label_page():
    return _page("label.html")


@app.get("/train")
def train_page():
    return _page("train.html")


@app.get("/test")
def test_page():
    return _page("test.html")





@app.get("/telemetry")
def telemetry_page():
    return _page("telemetry.html")


@app.get("/drive")
def drive_page():
    return _page("drive.html")


@app.get("/settings")
def settings_page():
    return _page("settings.html")


@app.get("/help")
def help_page():
    return _page("help.html")


# Serve shared assets (app.css, app.js) under /static
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
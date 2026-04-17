"""
ForzaTek AI — Database Layer
=============================
Single source of truth for the project's SQLite schema.

All other modules (recorder, ingester, labeler, trainer, predictor) talk to
the database through this module. Keeping schema in one place means adding
a column is a one-line change, not a hunt across 10 files.

Tables
------
frames         — raw captured frames (from live Forza or from videos)
labels         — one row per (frame, task) pair with provenance tracking
proposals      — model-generated label guesses awaiting review
models         — trained model checkpoints with their metadata
sources        — video files / URLs we've ingested, with HUD masks
active_queue   — which frames the active learner wants us to review next
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path("data/forzatek.db")

_write_lock = threading.Lock()  # SQLite can handle concurrent reads, not writes

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- ─────────── Frames: the raw data ───────────
CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    source_id       INTEGER,                       -- FK to sources (nullable for live)
    source_type     TEXT NOT NULL,                 -- 'live' | 'video'
    game_version    TEXT NOT NULL,                 -- 'fh4' | 'fh5' | 'fh6'
    biome           TEXT,
    weather         TEXT,
    time_of_day     TEXT,
    phash           INTEGER NOT NULL,
    frame_jpeg      BLOB NOT NULL,
    width           INTEGER,
    height          INTEGER,
    telemetry_json  TEXT,                          -- optional live-only
    video_time_sec  REAL,                          -- optional video-only
    label_status    TEXT DEFAULT 'unlabeled'       -- unlabeled|proposed|reviewed|labeled|skipped
);
CREATE INDEX IF NOT EXISTS idx_frame_status   ON frames(label_status);
CREATE INDEX IF NOT EXISTS idx_frame_version  ON frames(game_version);
CREATE INDEX IF NOT EXISTS idx_frame_bucket   ON frames(game_version, biome, weather, time_of_day);
CREATE INDEX IF NOT EXISTS idx_frame_source   ON frames(source_id);

-- ─────────── Labels: the ground truth ───────────
-- One row per (frame, task). task in {'seg', 'det', 'lane'}
-- data is JSON. Shape depends on task:
--   seg  -> {"mask_png_b64": "...", "classes": ["road","curb","wall","offroad"]}
--   det  -> {"boxes": [{"cls":"vehicle","x":0.1,"y":0.2,"w":0.1,"h":0.15}, ...]}
--   lane -> {"points": [[x0,y0,color],[x1,y1,color],...]}
CREATE TABLE IF NOT EXISTS labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    INTEGER NOT NULL,
    task        TEXT NOT NULL,
    data_json   TEXT NOT NULL,
    provenance  TEXT NOT NULL,    -- 'manual' | 'proposed_accepted' | 'proposed_edited' | 'auto_trusted'
    model_id    INTEGER,          -- which model made the proposal (null if manual)
    round_num   INTEGER DEFAULT 0,
    created_at  REAL NOT NULL,
    UNIQUE(frame_id, task),
    FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_labels_frame ON labels(frame_id);
CREATE INDEX IF NOT EXISTS idx_labels_task  ON labels(task);

-- ─────────── Proposals: model guesses awaiting review ───────────
-- Separate from labels so we never confuse "model said so" with "human confirmed"
CREATE TABLE IF NOT EXISTS proposals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL,
    task         TEXT NOT NULL,
    data_json    TEXT NOT NULL,
    confidence   REAL,
    uncertainty  REAL,              -- higher = model less sure (used by active learner)
    model_id     INTEGER,
    created_at   REAL NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_proposals_frame ON proposals(frame_id);
CREATE INDEX IF NOT EXISTS idx_proposals_unc   ON proposals(uncertainty DESC);

-- ─────────── Models: training checkpoints ───────────
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    round_num       INTEGER NOT NULL,
    path            TEXT NOT NULL,        -- relative path to checkpoint
    onnx_path       TEXT,
    trained_on      INTEGER NOT NULL,     -- number of frames in train set
    metrics_json    TEXT,                 -- {seg_iou, det_map, lane_err, ...}
    game_versions   TEXT,                 -- CSV of versions covered
    is_active       INTEGER DEFAULT 0,    -- 1 = current production model
    created_at      REAL NOT NULL
);

-- ─────────── Sources: video files and URLs we've ingested ───────────
CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,        -- 'video_file' | 'youtube_url'
    uri             TEXT NOT NULL,
    title           TEXT,
    game_version    TEXT,
    biome_override  TEXT,
    hud_mask_json   TEXT,                 -- [{x,y,w,h}, ...] normalized 0-1 coords
    duration_sec    REAL,
    frames_sampled  INTEGER DEFAULT 0,
    frames_accepted INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'pending',  -- pending|processing|done|failed
    created_at      REAL NOT NULL
);

-- ─────────── Active learning queue ───────────
CREATE TABLE IF NOT EXISTS active_queue (
    frame_id     INTEGER PRIMARY KEY,
    uncertainty  REAL NOT NULL,
    queued_at    REAL NOT NULL,
    round_num    INTEGER NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_queue_unc ON active_queue(uncertainty DESC);
"""


def init_db(db_path: Path = DB_PATH):
    """Create the database and all tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()
    conn.close()


@contextmanager
def read_conn(db_path: Path = DB_PATH):
    """Read-only connection context manager."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_conn(db_path: Path = DB_PATH):
    """Serialized write connection. Only one writer at a time."""
    with _write_lock:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


# ────────── Convenience helpers used across modules ──────────

def count_frames(db_path: Path = DB_PATH, **filters) -> int:
    with read_conn(db_path) as c:
        q = "SELECT COUNT(*) FROM frames"
        args = []
        if filters:
            clauses = []
            for k, v in filters.items():
                clauses.append(f"{k} = ?")
                args.append(v)
            q += " WHERE " + " AND ".join(clauses)
        return c.execute(q, args).fetchone()[0]


def count_labels(task: str = None, db_path: Path = DB_PATH) -> int:
    with read_conn(db_path) as c:
        if task:
            return c.execute("SELECT COUNT(*) FROM labels WHERE task=?", (task,)).fetchone()[0]
        return c.execute("SELECT COUNT(DISTINCT frame_id) FROM labels").fetchone()[0]


def get_active_model(db_path: Path = DB_PATH) -> dict | None:
    with read_conn(db_path) as c:
        r = c.execute("SELECT * FROM models WHERE is_active=1 LIMIT 1").fetchone()
        return dict(r) if r else None


def set_active_model(model_id: int, db_path: Path = DB_PATH):
    with write_conn(db_path) as c:
        c.execute("UPDATE models SET is_active=0")
        c.execute("UPDATE models SET is_active=1 WHERE id=?", (model_id,))


def overall_stats(db_path: Path = DB_PATH) -> dict:
    """One-shot dashboard summary."""
    with read_conn(db_path) as c:
        total = c.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        labeled = c.execute(
            "SELECT COUNT(DISTINCT frame_id) FROM labels WHERE task IN ('seg','det')"
        ).fetchone()[0]
        proposed = c.execute(
            "SELECT COUNT(DISTINCT frame_id) FROM proposals"
        ).fetchone()[0]
        queue = c.execute("SELECT COUNT(*) FROM active_queue").fetchone()[0]
        by_version = {}
        for row in c.execute("SELECT game_version, COUNT(*) FROM frames GROUP BY game_version"):
            by_version[row[0]] = row[1]
        active = c.execute("SELECT * FROM models WHERE is_active=1 LIMIT 1").fetchone()
        return {
            "total_frames": total,
            "labeled_frames": labeled,
            "proposed_frames": proposed,
            "queue_size": queue,
            "frames_by_version": by_version,
            "active_model": dict(active) if active else None,
        }


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
    print(json.dumps(overall_stats(), indent=2, default=str))

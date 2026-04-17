"""
ForzaTek AI — Labeling Backend
===============================
HTTP routes that power the labeling tool.

Key endpoints:
  GET  /api/label/next                Get next frame to label (respects queue)
  GET  /api/label/frame/{id}          Get a specific frame
  GET  /api/label/frame/{id}/image    Raw JPEG bytes
  POST /api/label/submit              Submit labels for a frame
  POST /api/label/skip                Mark frame as skipped
  GET  /api/label/prelabel/{id}       Generate pretrained-model proposals
  GET  /api/label/progress            Count stats

Frame selection priority:
  1. Frames in active_queue (picked by active learner), sorted by uncertainty
  2. Frames with status='proposed' (model made a guess, user should review)
  3. Frames with status='unlabeled' (fallback)
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

from backend.database import DB_PATH, init_db, read_conn, write_conn


def get_frame_jpeg(frame_id: int) -> bytes | None:
    with read_conn(DB_PATH) as c:
        r = c.execute("SELECT frame_jpeg FROM frames WHERE id=?", (frame_id,)).fetchone()
        return bytes(r["frame_jpeg"]) if r else None


def get_frame_row(frame_id: int) -> dict | None:
    with read_conn(DB_PATH) as c:
        r = c.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d.pop("frame_jpeg", None)
        return d


def get_existing_labels(frame_id: int) -> dict[str, dict]:
    with read_conn(DB_PATH) as c:
        rows = c.execute(
            "SELECT task, data_json, provenance FROM labels WHERE frame_id=?",
            (frame_id,),
        ).fetchall()
        return {
            r["task"]: {
                "data": json.loads(r["data_json"]),
                "provenance": r["provenance"],
            } for r in rows
        }


def get_existing_proposals(frame_id: int) -> dict[str, dict]:
    with read_conn(DB_PATH) as c:
        rows = c.execute(
            """SELECT task, data_json, confidence, uncertainty
               FROM proposals WHERE frame_id=?""",
            (frame_id,),
        ).fetchall()
        return {
            r["task"]: {
                "data": json.loads(r["data_json"]),
                "confidence": r["confidence"],
                "uncertainty": r["uncertainty"],
            } for r in rows
        }


def select_next_frame(skip_ids: list[int] = None) -> int | None:
    """
    Priority:
      1. active_queue entry with highest uncertainty
      2. frame with status='proposed' (newest first)
      3. frame with status='unlabeled' (oldest first)
    """
    skip_ids = skip_ids or []
    placeholder = ",".join("?" for _ in skip_ids) if skip_ids else "NULL"
    with read_conn(DB_PATH) as c:
        # 1. Active queue
        q = f"""SELECT frame_id FROM active_queue
                WHERE frame_id NOT IN ({placeholder})
                ORDER BY uncertainty DESC LIMIT 1"""
        r = c.execute(q, skip_ids).fetchone()
        if r:
            return r["frame_id"]
        # 2. Proposed
        q = f"""SELECT id FROM frames
                WHERE label_status='proposed' AND id NOT IN ({placeholder})
                ORDER BY id DESC LIMIT 1"""
        r = c.execute(q, skip_ids).fetchone()
        if r:
            return r["id"]
        # 3. Unlabeled
        q = f"""SELECT id FROM frames
                WHERE label_status='unlabeled' AND id NOT IN ({placeholder})
                ORDER BY RANDOM() LIMIT 1"""
        r = c.execute(q, skip_ids).fetchone()
        if r:
            return r["id"]
    return None


def submit_labels(frame_id: int, labels: dict, round_num: int = 0):
    """
    labels: {
      'seg': {'data': {...}, 'provenance': 'manual'|'proposed_accepted'|'proposed_edited'},
      'det': {...},
      'lane': {...} (optional)
    }
    """
    now = time.time()
    with write_conn(DB_PATH) as c:
        for task, payload in labels.items():
            data = payload.get("data")
            prov = payload.get("provenance", "manual")
            if data is None:
                continue
            # Upsert
            c.execute(
                """INSERT INTO labels
                   (frame_id, task, data_json, provenance, round_num, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(frame_id, task) DO UPDATE SET
                     data_json=excluded.data_json,
                     provenance=excluded.provenance,
                     round_num=excluded.round_num,
                     created_at=excluded.created_at""",
                (frame_id, task, json.dumps(data), prov, round_num, now)
            )
        c.execute("UPDATE frames SET label_status='labeled' WHERE id=?", (frame_id,))
        c.execute("DELETE FROM active_queue WHERE frame_id=?", (frame_id,))


def skip_frame(frame_id: int):
    with write_conn(DB_PATH) as c:
        c.execute("UPDATE frames SET label_status='skipped' WHERE id=?", (frame_id,))
        c.execute("DELETE FROM active_queue WHERE frame_id=?", (frame_id,))


def labeling_progress() -> dict:
    with read_conn(DB_PATH) as c:
        total = c.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        labeled = c.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='labeled'"
        ).fetchone()[0]
        proposed = c.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='proposed'"
        ).fetchone()[0]
        unlabeled = c.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='unlabeled'"
        ).fetchone()[0]
        skipped = c.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='skipped'"
        ).fetchone()[0]
        queue = c.execute("SELECT COUNT(*) FROM active_queue").fetchone()[0]
        # Per task
        seg_count = c.execute("SELECT COUNT(*) FROM labels WHERE task='seg'").fetchone()[0]
        det_count = c.execute("SELECT COUNT(*) FROM labels WHERE task='det'").fetchone()[0]
        return {
            "total": total, "labeled": labeled, "proposed": proposed,
            "unlabeled": unlabeled, "skipped": skipped, "queue": queue,
            "seg_labels": seg_count, "det_labels": det_count,
        }


def register_routes(app):
    from fastapi import Body, HTTPException, Response
    from fastapi.responses import JSONResponse

    @app.get("/api/label/progress")
    def _progress():
        return labeling_progress()

    @app.get("/api/label/next")
    def _next():
        fid = select_next_frame()
        if fid is None:
            return {"frame_id": None}
        row = get_frame_row(fid)
        return {
            "frame_id": fid,
            "frame": row,
            "labels": get_existing_labels(fid),
            "proposals": get_existing_proposals(fid),
        }

    @app.get("/api/label/frame/{frame_id}")
    def _get_frame(frame_id: int):
        row = get_frame_row(frame_id)
        if not row:
            raise HTTPException(404, "frame not found")
        return {
            "frame": row,
            "labels": get_existing_labels(frame_id),
            "proposals": get_existing_proposals(frame_id),
        }

    @app.get("/api/label/frame/{frame_id}/image")
    def _get_image(frame_id: int):
        jpg = get_frame_jpeg(frame_id)
        if not jpg:
            raise HTTPException(404)
        return Response(content=jpg, media_type="image/jpeg")

    @app.post("/api/label/submit")
    def _submit(payload: dict = Body(...)):
        fid = int(payload["frame_id"])
        submit_labels(fid, payload.get("labels", {}), int(payload.get("round_num", 0)))
        return {"ok": True, "progress": labeling_progress()}

    @app.post("/api/label/skip")
    def _skip(payload: dict = Body(...)):
        skip_frame(int(payload["frame_id"]))
        return {"ok": True}

    @app.get("/api/label/prelabel/{frame_id}")
    def _prelabel(frame_id: int):
        jpg = get_frame_jpeg(frame_id)
        if not jpg:
            raise HTTPException(404)
        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        try:
            from backend.prelabeler import prelabel_both
            result = prelabel_both(frame)
            return result
        except Exception as e:
            raise HTTPException(500, f"prelabel failed: {e}")

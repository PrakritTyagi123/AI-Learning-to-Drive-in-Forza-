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
    with read_conn(DB_PATH) as c:
        if skip_ids:
            ph = ",".join("?" for _ in skip_ids)
            skip_q  = f"AND frame_id NOT IN ({ph})"
            skip_id = f"AND id NOT IN ({ph})"
            args = skip_ids
        else:
            skip_q = skip_id = ""
            args = []
        r = c.execute(f"SELECT frame_id FROM active_queue WHERE 1=1 {skip_q} ORDER BY uncertainty DESC LIMIT 1", args).fetchone()
        if r: return r["frame_id"]
        r = c.execute(f"SELECT id FROM frames WHERE label_status='proposed' {skip_id} ORDER BY id DESC LIMIT 1", args).fetchone()
        if r: return r["id"]
        r = c.execute(f"SELECT id FROM frames WHERE label_status='unlabeled' {skip_id} ORDER BY RANDOM() LIMIT 1", args).fetchone()
        if r: return r["id"]
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



# ─── Bulk pre-label job ───
import threading as _threading

_bulk_state = {"running": False, "total": 0, "done": 0, "failed": 0, "cancelled": False, "error": ""}
_bulk_lock = _threading.Lock()


def _bulk_worker(include_proposed: bool = False):
    import json as _json
    import numpy as _np
    from backend.prelabeler import prelabel_both
    with write_conn(DB_PATH) as c:
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_proposals_frame_task ON proposals(frame_id, task)")
    with read_conn(DB_PATH) as c:
        if include_proposed:
            rows = c.execute("SELECT id FROM frames WHERE label_status IN ('unlabeled','proposed') ORDER BY id").fetchall()
        else:
            rows = c.execute("SELECT id FROM frames WHERE label_status='unlabeled' ORDER BY id").fetchall()
    frame_ids = [r["id"] for r in rows]
    with _bulk_lock:
        _bulk_state["total"] = len(frame_ids)
        _bulk_state["done"] = 0
        _bulk_state["failed"] = 0
        _bulk_state["error"] = ""
    for fid in frame_ids:
        with _bulk_lock:
            if _bulk_state["cancelled"]: break
        jpg = get_frame_jpeg(fid)
        if not jpg:
            with _bulk_lock: _bulk_state["failed"] += 1
            continue
        try:
            arr = _np.frombuffer(jpg, dtype=_np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            result = prelabel_both(frame)
            now = time.time()
            seg = result.get("seg", {})
            det = result.get("det", {})
            with write_conn(DB_PATH) as c:
                # Clear old proposals first
                c.execute("DELETE FROM proposals WHERE frame_id=?", (fid,))
                if seg.get("mask_png_b64"):
                    c.execute("""INSERT INTO proposals (frame_id, task, data_json, confidence, uncertainty, created_at) VALUES (?, 'seg', ?, 0.7, 0.3, ?)""",
                              (fid, _json.dumps(seg), now))
                if det:
                    c.execute("""INSERT INTO proposals (frame_id, task, data_json, confidence, uncertainty, created_at) VALUES (?, 'det', ?, 0.7, 0.3, ?)""",
                              (fid, _json.dumps(det), now))
                c.execute("UPDATE frames SET label_status='proposed' WHERE id=?", (fid,))
            with _bulk_lock: _bulk_state["done"] += 1
        except Exception as ex:
            with _bulk_lock:
                _bulk_state["failed"] += 1
                _bulk_state["error"] = str(ex)
    with _bulk_lock:
        _bulk_state["running"] = False



# ─── Auto-accept high-confidence proposals ───

_auto_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "accepted": 0,
    "skipped": 0,
    "cancelled": False,
    "error": "",
}
_auto_lock = _threading.Lock()


def _score_proposal(seg_mask_png_b64: str, det_data: dict) -> float:
    """Compute a 'confidence' score for a proposal. Higher = more likely correct.

    Heuristics (these are rough but effective):
      - Seg: mean confidence not available, so use % of frame that is road. A very
        tiny or very huge road area is suspicious (intro/menu or bad mask).
        A 15-60% road coverage with a largest-component close to the full mask
        means a clean confident prediction.
      - Det: average box confidence + penalty for very small boxes or
        non-bottom-half boxes (they're usually false positives in Forza).
    """
    import base64 as _b64
    import numpy as _np
    import cv2 as _cv2

    score_seg = 0.0
    try:
        raw = _b64.b64decode(seg_mask_png_b64)
        arr = _np.frombuffer(raw, dtype=_np.uint8)
        mask = _cv2.imdecode(arr, _cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            H, W = mask.shape
            road_pct = float((mask == 1).sum()) / float(H * W)
            # Preferred range: 15-60% of frame is road
            if 0.12 <= road_pct <= 0.65:
                score_seg = 1.0
            elif 0.05 <= road_pct < 0.12 or 0.65 < road_pct <= 0.80:
                score_seg = 0.6
            else:
                score_seg = 0.2
            # Bonus: connected-component cohesiveness
            n, labels, stats, _ = _cv2.connectedComponentsWithStats((mask == 1).astype(_np.uint8))
            if n > 1:
                biggest_frac = stats[1:, _cv2.CC_STAT_AREA].max() / float(max((mask == 1).sum(), 1))
                if biggest_frac > 0.85:
                    score_seg = min(1.0, score_seg + 0.1)
    except Exception:
        score_seg = 0.0

    score_det = 1.0  # default: no boxes is fine (no false positives)
    try:
        boxes = det_data.get("boxes", []) if isinstance(det_data, dict) else []
        if boxes:
            confs = [float(b.get("confidence", 0.0)) for b in boxes]
            if confs:
                avg = sum(confs) / len(confs)
                # Penalize if any box is too small or in the top third of frame
                bad = 0
                for b in boxes:
                    if b.get("h", 0) < 0.03 or b.get("w", 0) < 0.03:
                        bad += 1
                    if b.get("y", 1.0) + b.get("h", 0) / 2 < 0.30:
                        bad += 1
                if bad > 0:
                    avg *= max(0.3, 1.0 - 0.2 * bad)
                score_det = max(0.0, min(1.0, avg))

    except Exception:
        score_det = 0.5

    # Combined score: both must be reasonable
    return min(score_seg, score_det)


def _auto_accept_worker(threshold: float):
    import json as _json
    try:
        with read_conn(DB_PATH) as c:
            rows = c.execute(
                """SELECT f.id as frame_id,
                          MAX(CASE WHEN p.task='seg' THEN p.data_json END) AS seg_json,
                          MAX(CASE WHEN p.task='det' THEN p.data_json END) AS det_json
                   FROM frames f
                   JOIN proposals p ON p.frame_id = f.id
                   WHERE f.label_status='proposed'
                   GROUP BY f.id"""
            ).fetchall()

        with _auto_lock:
            _auto_state["total"] = len(rows)
            _auto_state["done"] = 0
            _auto_state["accepted"] = 0
            _auto_state["skipped"] = 0
            _auto_state["error"] = ""

        for row in rows:
            with _auto_lock:
                if _auto_state["cancelled"]:
                    break

            fid = row["frame_id"]
            seg_data = {}
            det_data = {}
            try:
                if row["seg_json"]:
                    seg_data = _json.loads(row["seg_json"])
                if row["det_json"]:
                    det_data = _json.loads(row["det_json"])
            except Exception:
                with _auto_lock:
                    _auto_state["done"] += 1
                    _auto_state["skipped"] += 1
                continue

            seg_b64 = (seg_data or {}).get("mask_png_b64", "")
            score = _score_proposal(seg_b64, det_data or {})

            if score >= threshold:
                # Accept: copy proposals -> labels
                now = time.time()
                try:
                    with write_conn(DB_PATH) as c:
                        if seg_data:
                            c.execute(
                                """INSERT INTO labels (frame_id, task, data_json, provenance, round_num, created_at)
                                   VALUES (?, 'seg', ?, 'proposed_accepted', 0, ?)
                                   ON CONFLICT(frame_id, task) DO UPDATE SET
                                     data_json=excluded.data_json, provenance=excluded.provenance,
                                     created_at=excluded.created_at""",
                                (fid, _json.dumps(seg_data), now),
                            )
                        if det_data:
                            c.execute(
                                """INSERT INTO labels (frame_id, task, data_json, provenance, round_num, created_at)
                                   VALUES (?, 'det', ?, 'proposed_accepted', 0, ?)
                                   ON CONFLICT(frame_id, task) DO UPDATE SET
                                     data_json=excluded.data_json, provenance=excluded.provenance,
                                     created_at=excluded.created_at""",
                                (fid, _json.dumps(det_data), now),
                            )
                        c.execute("UPDATE frames SET label_status='labeled' WHERE id=?", (fid,))
                    with _auto_lock:
                        _auto_state["accepted"] += 1
                except Exception as ex:
                    with _auto_lock:
                        _auto_state["error"] = str(ex)[:300]
                        _auto_state["skipped"] += 1
            else:
                with _auto_lock:
                    _auto_state["skipped"] += 1

            with _auto_lock:
                _auto_state["done"] += 1

    except Exception as ex:
        with _auto_lock:
            _auto_state["error"] = str(ex)[:500]

    with _auto_lock:
        _auto_state["running"] = False


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

    @app.post("/api/label/prelabel_all")
    def _prelabel_all(payload: dict = Body(default={})):
        include_proposed = bool(payload.get("include_proposed", False))
        with _bulk_lock:
            if _bulk_state["running"]:
                return {"ok": False, "reason": "already running"}
            _bulk_state["running"] = True
            _bulk_state["cancelled"] = False
        _threading.Thread(target=_bulk_worker, args=(include_proposed,), daemon=True).start()
        return {"ok": True}

    @app.post("/api/label/prelabel_all/cancel")
    def _prelabel_cancel():
        with _bulk_lock: _bulk_state["cancelled"] = True
        return {"ok": True}

    @app.get("/api/label/prelabel_all/status")
    def _prelabel_status():
        with _bulk_lock: return dict(_bulk_state)

    @app.post("/api/label/auto_accept")
    def _auto_accept(payload: dict = Body(default={})):
        threshold = float(payload.get("threshold", 0.7))
        with _auto_lock:
            if _auto_state["running"]:
                return {"ok": False, "reason": "already running"}
            _auto_state["running"] = True
            _auto_state["cancelled"] = False
        _threading.Thread(target=_auto_accept_worker, args=(threshold,), daemon=True).start()
        return {"ok": True, "threshold": threshold}

    @app.post("/api/label/auto_accept/cancel")
    def _auto_accept_cancel():
        with _auto_lock: _auto_state["cancelled"] = True
        return {"ok": True}

    @app.get("/api/label/auto_accept/status")
    def _auto_accept_status():
        with _auto_lock: return dict(_auto_state)

    @app.post("/api/label/reset_proposed")
    def _reset_proposed():
        """Reset all proposed frames back to unlabeled and clear their proposals."""
        with write_conn(DB_PATH) as c:
            c.execute("DELETE FROM proposals WHERE frame_id IN (SELECT id FROM frames WHERE label_status='proposed')")
            res = c.execute("UPDATE frames SET label_status='unlabeled' WHERE label_status='proposed'")
        return {"ok": True}
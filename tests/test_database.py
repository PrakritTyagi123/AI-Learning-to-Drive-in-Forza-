"""
Tests for the database layer.
Run with: python -m tests.test_database
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import (
    init_db, write_conn, read_conn,
    count_frames, count_labels, get_active_model, set_active_model,
    overall_stats,
)


def test_init_creates_tables():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        init_db(db)
        with read_conn(db) as c:
            tables = {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        expected = {"meta", "frames", "labels", "proposals", "models", "sources", "active_queue"}
        assert expected.issubset(tables), f"missing tables: {expected - tables}"
    print("✓ init_creates_tables")


def test_frame_insert_and_count():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        init_db(db)
        with write_conn(db) as c:
            for i in range(5):
                c.execute(
                    """INSERT INTO frames
                       (ts, source_type, game_version, phash, frame_jpeg)
                       VALUES (?, 'live', 'fh4', ?, ?)""",
                    (time.time(), i, b"fake_jpeg_bytes")
                )
        assert count_frames(db) == 5
        assert count_frames(db, game_version="fh4") == 5
        assert count_frames(db, game_version="fh5") == 0
    print("✓ frame_insert_and_count")


def test_label_upsert():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        init_db(db)
        with write_conn(db) as c:
            cur = c.execute(
                """INSERT INTO frames (ts, source_type, game_version, phash, frame_jpeg)
                   VALUES (?, 'live', 'fh4', 0, ?)""",
                (time.time(), b"x"))
            frame_id = cur.lastrowid
            c.execute(
                """INSERT INTO labels (frame_id, task, data_json, provenance, created_at)
                   VALUES (?, 'seg', ?, 'manual', ?)""",
                (frame_id, json.dumps({"v": 1}), time.time()))
            # Upsert same (frame, task) - should update, not duplicate
            c.execute(
                """INSERT INTO labels (frame_id, task, data_json, provenance, created_at)
                   VALUES (?, 'seg', ?, 'proposed_edited', ?)
                   ON CONFLICT(frame_id, task) DO UPDATE SET
                     data_json=excluded.data_json,
                     provenance=excluded.provenance""",
                (frame_id, json.dumps({"v": 2}), time.time()))
        with read_conn(db) as c:
            rows = c.execute("SELECT * FROM labels WHERE frame_id=?", (frame_id,)).fetchall()
        assert len(rows) == 1, "upsert should not create duplicate rows"
        assert json.loads(rows[0]["data_json"])["v"] == 2
        assert rows[0]["provenance"] == "proposed_edited"
    print("✓ label_upsert")


def test_active_model_toggle():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        init_db(db)
        with write_conn(db) as c:
            for i in range(3):
                c.execute(
                    """INSERT INTO models (name, round_num, path, trained_on, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f"model_{i}", i, f"models/{i}.pt", 100, time.time()))
        assert get_active_model(db) is None
        set_active_model(2, db)
        active = get_active_model(db)
        assert active is not None and active["id"] == 2
        set_active_model(3, db)
        active = get_active_model(db)
        assert active["id"] == 3  # only one active at a time
    print("✓ active_model_toggle")


def test_overall_stats_shape():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        init_db(db)
        s = overall_stats(db)
        required = {"total_frames", "labeled_frames", "proposed_frames",
                    "queue_size", "frames_by_version", "active_model"}
        assert required.issubset(s.keys())
        assert s["total_frames"] == 0
    print("✓ overall_stats_shape")


if __name__ == "__main__":
    test_init_creates_tables()
    test_frame_insert_and_count()
    test_label_upsert()
    test_active_model_toggle()
    test_overall_stats_shape()
    print("\nAll database tests passed.")

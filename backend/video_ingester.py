"""
ForzaTek AI — Video Ingester
=============================
Pulls frames from YouTube URLs or local video files into the database.

Pipeline per video:
  1. Download (if URL) or locate (if local file)
  2. Sample frames at configured rate
  3. Apply HUD mask (user-defined rectangle(s) blanked out)
  4. Skip non-gameplay frames (menus, black frames, static scenes)
  5. Perceptual-hash dedup
  6. Auto-tag biome/weather/time-of-day
  7. Insert into frames table
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Video decoding backends (in priority order)
# 1. PyAV with NVDEC hardware decoder (NVIDIA GPU, fastest on Windows)
# 2. decord with CPU threads (fast CPU multi-threaded fallback)
# 3. cv2.VideoCapture (last resort)
try:
    import av
    PYAV_AVAILABLE = True
except Exception:
    PYAV_AVAILABLE = False

try:
    import decord
    DECORD_AVAILABLE = True
except Exception:
    DECORD_AVAILABLE = False

from backend.database import DB_PATH, init_db, write_conn, read_conn
from backend.recorder import (
    compute_phash, hamming_distance,
    infer_weather, infer_time_of_day,
    PHASH_HAMMING_THRESHOLD, PHASH_WINDOW_SIZE, JPEG_QUALITY,
)


# ────────── Menu / intro / non-gameplay detection ──────────

def looks_like_non_gameplay(frame_bgr: np.ndarray) -> str | None:
    """Return a reason string if this frame should be rejected, else None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    std = gray.std()

    # Fully black / dark transition
    if mean < 15:
        return "too_dark"
    # Fully white (compression artifact / loading screen)
    if mean > 245:
        return "too_bright"
    # Very low variance = flat color = menu or loading
    if std < 12:
        return "low_variance"
    # Huge solid color regions in the center = UI panels
    h, w = gray.shape
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if center.std() < 8:
        return "solid_center"
    return None


# ────────── HUD masking ──────────

def apply_hud_mask(frame_bgr: np.ndarray, mask_rects: list[dict]) -> np.ndarray:
    """
    mask_rects: list of {x,y,w,h} with coords normalized 0-1.
    Blacks out those regions. Use black (not crop) so resolution stays constant.
    """
    if not mask_rects:
        return frame_bgr
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    for r in mask_rects:
        x1 = int(r["x"] * W)
        y1 = int(r["y"] * H)
        x2 = int((r["x"] + r["w"]) * W)
        y2 = int((r["y"] + r["h"]) * H)
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = 0
    return out


# ────────── yt-dlp download wrapper ──────────

def download_with_ytdlp(url: str, out_dir: Path, max_height: int = 1080) -> Path:
    """
    Downloads the best video track up to max_height. Returns path to the file.
    Requires yt-dlp and ffmpeg installed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={max_height}][ext=mp4]/best[height<={max_height}]",
        "--no-playlist",
        "-o", out_template,
        "--print", "after_move:filepath",
        "--no-warnings",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {proc.stderr[:500]}")
    # Last non-empty line of stdout is the filepath
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("yt-dlp produced no filepath output")
    return Path(lines[-1])


def get_video_title(url: str) -> str:
    """Fetch just the title via yt-dlp --get-title (fast, no download)."""
    try:
        proc = subprocess.run(
            ["yt-dlp", "--get-title", "--no-warnings", url],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return url


# ────────── The ingester ──────────

@dataclass
class IngestJob:
    source_id: int
    title: str
    game_version: str
    biome_override: Optional[str]
    sample_every_sec: float
    hud_mask: list
    status: str = "pending"           # pending | downloading | processing | done | failed
    error: str = ""
    frames_sampled: int = 0
    frames_accepted: int = 0
    frames_rejected: dict = field(default_factory=lambda: {
        "dedup": 0, "non_gameplay": 0, "encode_fail": 0,
    })
    started_at: float = 0
    finished_at: float = 0
    current_video_time: float = 0
    total_duration: float = 0

    def snapshot(self) -> dict:
        d = dict(self.__dict__)
        d["progress"] = (self.current_video_time / self.total_duration) if self.total_duration else 0
        return d


class VideoIngester:
    """Runs one ingest job at a time in a background thread."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_db(db_path)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._current: Optional[IngestJob] = None
        self._cancel_flag = False
        self._download_dir = Path("data/videos")
        self._download_dir.mkdir(parents=True, exist_ok=True)

    # ─── registering a source ───

    def register_source(self, kind: str, uri: str, game_version: str,
                        biome_override: Optional[str], hud_mask: list,
                        title: Optional[str] = None) -> int:
        with write_conn(self.db_path) as c:
            cur = c.execute(
                """INSERT INTO sources
                   (kind, uri, title, game_version, biome_override,
                    hud_mask_json, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (kind, uri, title or uri, game_version, biome_override,
                 json.dumps(hud_mask), time.time())
            )
            return cur.lastrowid

    def list_sources(self) -> list[dict]:
        with read_conn(self.db_path) as c:
            rows = c.execute("SELECT * FROM sources ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    # ─── submitting a job ───

    def submit(self, source_id: int, sample_every_sec: float = 1.0) -> bool:
        """Kick off ingest in background. Returns False if another job is running."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._cancel_flag = False
            with read_conn(self.db_path) as c:
                src = c.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
            if not src:
                return False
            hud_mask = json.loads(src["hud_mask_json"] or "[]")
            self._current = IngestJob(
                source_id=source_id,
                title=src["title"],
                game_version=src["game_version"],
                biome_override=src["biome_override"],
                sample_every_sec=sample_every_sec,
                hud_mask=hud_mask,
                started_at=time.time(),
            )
            self._thread = threading.Thread(
                target=self._run, args=(src["kind"], src["uri"]), daemon=True,
            )
            self._thread.start()
        return True

    def cancel(self):
        with self._lock:
            self._cancel_flag = True

    def status(self) -> Optional[dict]:
        with self._lock:
            return self._current.snapshot() if self._current else None

    # ─── the actual ingest ───

    def _run(self, kind: str, uri: str):
        job = self._current
        try:
            # Step 1: locate/download video
            if kind == "youtube_url":
                job.status = "downloading"
                video_path = download_with_ytdlp(uri, self._download_dir)
            else:
                video_path = Path(uri)
                if not video_path.exists():
                    raise FileNotFoundError(f"video file not found: {uri}")

            # Step 2: open the video with the fastest available backend
            backend = None           # "pyav_gpu" | "pyav_cpu" | "decord" | "cv2"
            container = None         # for pyav
            stream = None             # for pyav
            vr = None                 # for decord
            cap = None                # for cv2

            # Try PyAV with NVDEC first (true GPU hardware decode)
            if PYAV_AVAILABLE:
                try:
                    container = av.open(str(video_path), options={
                        "hwaccel": "cuda",
                        "hwaccel_output_format": "cuda",
                    })
                    stream = container.streams.video[0]
                    stream.codec_context.options = {"hwaccel": "cuda"}
                    # Set threading for faster decode
                    stream.thread_type = "AUTO"
                    backend = "pyav_gpu"
                    print(f"[INGEST] Using PyAV with NVDEC (GPU hardware decode) for {video_path.name}")
                except Exception as e:
                    print(f"[INGEST] PyAV NVDEC failed ({e}), trying PyAV CPU")
                    try:
                        if container is not None:
                            container.close()
                    except Exception:
                        pass
                    container = None
                    try:
                        container = av.open(str(video_path))
                        stream = container.streams.video[0]
                        stream.thread_type = "AUTO"
                        backend = "pyav_cpu"
                        print(f"[INGEST] Using PyAV with CPU threads for {video_path.name}")
                    except Exception as e2:
                        print(f"[INGEST] PyAV CPU also failed ({e2})")
                        container = None
                        stream = None

            # Try decord next
            if backend is None and DECORD_AVAILABLE:
                try:
                    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=4)
                    backend = "decord"
                    print(f"[INGEST] Using decord with CPU threads for {video_path.name}")
                except Exception as e:
                    print(f"[INGEST] decord failed ({e})")
                    vr = None

            # Last resort: cv2
            if backend is None:
                print(f"[INGEST] Using cv2.VideoCapture (slowest) for {video_path.name}")
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError("cv2 could not open the video")
                backend = "cv2"

            # Get fps and total_frames based on backend
            if backend.startswith("pyav"):
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                # total_frames may be inaccurate in some containers; duration * fps is more reliable
                if stream.frames and stream.frames > 0:
                    total_frames = stream.frames
                elif stream.duration and stream.time_base:
                    total_frames = int(float(stream.duration * stream.time_base) * fps)
                else:
                    total_frames = 10**9  # sentinel: read until EOF
            elif backend == "decord":
                fps = vr.get_avg_fps() or 30.0
                total_frames = len(vr)
            else:  # cv2
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            job.total_duration = total_frames / fps if fps else 0
            frame_interval = int(round(fps * job.sample_every_sec))
            if frame_interval < 1:
                frame_interval = 1

            with write_conn(self.db_path) as c:
                c.execute(
                    "UPDATE sources SET duration_sec=?, status='processing' WHERE id=?",
                    (job.total_duration, job.source_id),
                )

            job.status = "processing"
            recent_hashes: list[int] = []
            frame_idx = 0

            while True:
                if self._cancel_flag:
                    job.status = "cancelled"
                    break

                if frame_idx >= total_frames:
                    break

                # Read one frame at frame_idx based on backend
                frame = None
                try:
                    if backend.startswith("pyav"):
                        # Seek to timestamp
                        target_ts = int(frame_idx / fps / stream.time_base) if stream.time_base else 0
                        try:
                            container.seek(target_ts, backward=True, any_frame=False, stream=stream)
                        except Exception:
                            pass
                        # Decode frames after seek until we find one at/past target
                        want_pts = frame_idx / fps
                        got = None
                        for packet in container.demux(stream):
                            for f in packet.decode():
                                if f is None:
                                    continue
                                pts = float(f.pts * stream.time_base) if f.pts is not None else 0
                                if pts >= want_pts - (0.5 / fps):
                                    got = f
                                    break
                            if got is not None:
                                break
                        if got is None:
                            break
                        # PyAV: convert to ndarray BGR
                        arr = got.to_ndarray(format="bgr24")
                        frame = arr
                    elif backend == "decord":
                        f = vr[frame_idx]
                        arr = f.asnumpy() if hasattr(f, "asnumpy") else np.asarray(f)
                        frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    else:  # cv2
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret:
                            break
                except Exception as e:
                    print(f"[INGEST] frame read error at {frame_idx}: {e}")
                    break

                if frame is None:
                    break

                job.current_video_time = frame_idx / fps
                job.frames_sampled += 1

                # HUD mask
                frame_masked = apply_hud_mask(frame, job.hud_mask)

                # Menu/non-gameplay filter (on the masked frame)
                reason = looks_like_non_gameplay(frame_masked)
                if reason:
                    job.frames_rejected["non_gameplay"] += 1
                    frame_idx += frame_interval
                    continue

                # pHash dedup
                ph = compute_phash(frame_masked) & 0x7FFFFFFFFFFFFFFF
                dup = False
                for h in recent_hashes:
                    if hamming_distance(ph, h) <= PHASH_HAMMING_THRESHOLD:
                        dup = True
                        break
                if dup:
                    job.frames_rejected["dedup"] += 1
                    frame_idx += frame_interval
                    continue

                # Encode & save
                ok, buf = cv2.imencode(".jpg", frame_masked, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    job.frames_rejected["encode_fail"] += 1
                    frame_idx += frame_interval
                    continue

                biome = job.biome_override or "unknown"
                weather = infer_weather(frame_masked)
                tod = infer_time_of_day(frame_masked)
                h_px, w_px = frame_masked.shape[:2]

                with write_conn(self.db_path) as c:
                    c.execute(
                        """INSERT INTO frames
                           (ts, source_id, source_type, game_version, biome, weather,
                            time_of_day, phash, frame_jpeg, width, height, video_time_sec)
                           VALUES (?, ?, 'video', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (time.time(), job.source_id, job.game_version, biome,
                         weather, tod, ph, buf.tobytes(), w_px, h_px,
                         job.current_video_time)
                    )

                recent_hashes.append(ph)
                if len(recent_hashes) > PHASH_WINDOW_SIZE:
                    recent_hashes.pop(0)
                job.frames_accepted += 1
                frame_idx += frame_interval

            try:
                if backend.startswith("pyav") and container is not None:
                    container.close()
                elif backend == "cv2" and cap is not None:
                    cap.release()
            except Exception:
                pass

            with write_conn(self.db_path) as c:
                c.execute(
                    """UPDATE sources SET status=?, frames_sampled=?, frames_accepted=?
                       WHERE id=?""",
                    ("done" if not self._cancel_flag else "cancelled",
                     job.frames_sampled, job.frames_accepted, job.source_id)
                )

            if not self._cancel_flag:
                job.status = "done"
            job.finished_at = time.time()

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.finished_at = time.time()
            with write_conn(self.db_path) as c:
                c.execute("UPDATE sources SET status='failed' WHERE id=?", (job.source_id,))


ingester = VideoIngester()



# ─── Background YouTube download queue ───
# Instead of blocking HTTP requests for each download, the frontend POSTs a list
# of URLs and a background worker downloads them one-by-one, updating status as
# it goes. This way the UI never gets stuck.

import threading as _dl_threading

_dl_queue = []                      # list of {url, status, title, path, error}
_dl_thread = None
_dl_lock = _dl_threading.Lock()
_dl_current = None                  # dict of the item currently downloading (or None)
_dl_cancel = False


def _dl_worker():
    """Download items from the queue one at a time."""
    global _dl_current, _dl_cancel
    while True:
        with _dl_lock:
            # find first pending item
            item = None
            for it in _dl_queue:
                if it["status"] == "pending":
                    item = it
                    break
            if item is None or _dl_cancel:
                _dl_current = None
                break
            item["status"] = "downloading"
            _dl_current = item

        try:
            path = download_with_ytdlp(item["url"], ingester._download_dir)
            title = get_video_title(item["url"])
            with _dl_lock:
                item["status"] = "done"
                item["path"] = str(path)
                item["title"] = title

            # Auto-register as a source
            try:
                ingester.register_source(
                    kind="video_file",
                    uri=str(path),
                    game_version=item.get("game_version", "fh4"),
                    biome_override=item.get("biome_override"),
                    hud_mask=[],
                    title=title,
                )
            except Exception as reg_e:
                with _dl_lock:
                    item["error"] = f"Downloaded but register failed: {reg_e}"
        except Exception as e:
            with _dl_lock:
                item["status"] = "failed"
                item["error"] = str(e)[:500]

    with _dl_lock:
        _dl_current = None


def _ensure_worker():
    """Start the worker thread if it isn't already running."""
    global _dl_thread
    with _dl_lock:
        if _dl_thread is None or not _dl_thread.is_alive():
            _dl_thread = _dl_threading.Thread(target=_dl_worker, daemon=True)
            _dl_thread.start()


def register_routes(app):
    from fastapi import Body, UploadFile, File, HTTPException

    @app.get("/api/ingest/sources")
    def _sources():
        return {"sources": ingester.list_sources()}

    @app.post("/api/ingest/download_youtube")
    def _download_youtube(payload: dict = Body(...)):
        """Download a YouTube URL now (synchronous) and return the local file path."""
        url = (payload.get("url") or "").strip()
        if not url:
            raise HTTPException(400, "url required")
        try:
            path = download_with_ytdlp(url, ingester._download_dir)
            return {"ok": True, "path": str(path), "title": get_video_title(url)}
        except Exception as e:
            raise HTTPException(500, f"Download failed: {e}")

    @app.post("/api/ingest/download_queue")
    def _download_queue(payload: dict = Body(...)):
        """Queue URLs for background download. Returns immediately."""
        urls = payload.get("urls") or []
        if isinstance(urls, str):
            urls = [u.strip() for u in urls.splitlines() if u.strip()]
        if not urls:
            raise HTTPException(400, "no urls provided")
        game_version = payload.get("game_version", "fh4")
        biome_override = payload.get("biome_override")
        global _dl_cancel
        with _dl_lock:
            _dl_cancel = False
            for url in urls:
                # Skip duplicates already in queue or done
                if any(it["url"] == url for it in _dl_queue):
                    continue
                _dl_queue.append({
                    "url": url,
                    "status": "pending",
                    "title": "",
                    "path": "",
                    "error": "",
                    "game_version": game_version,
                    "biome_override": biome_override,
                })
        _ensure_worker()
        return {"ok": True, "queued": len(urls)}

    @app.get("/api/ingest/download_queue")
    def _download_queue_status():
        """Return the current state of the queue."""
        with _dl_lock:
            items = [
                {
                    "url": it["url"],
                    "status": it["status"],
                    "title": it.get("title", ""),
                    "path": it.get("path", ""),
                    "error": it.get("error", ""),
                } for it in _dl_queue
            ]
        pending  = sum(1 for it in items if it["status"] == "pending")
        active   = sum(1 for it in items if it["status"] == "downloading")
        done     = sum(1 for it in items if it["status"] == "done")
        failed   = sum(1 for it in items if it["status"] == "failed")
        return {
            "items": items,
            "pending": pending,
            "active": active,
            "done": done,
            "failed": failed,
            "running": active > 0 or pending > 0,
        }

    @app.post("/api/ingest/download_queue/clear")
    def _download_queue_clear():
        """Clear completed/failed entries from the queue."""
        with _dl_lock:
            _dl_queue[:] = [it for it in _dl_queue if it["status"] in ("pending", "downloading")]
        return {"ok": True}

    @app.post("/api/ingest/download_queue/cancel")
    def _download_queue_cancel():
        """Cancel any pending downloads (doesn't interrupt the active one)."""
        global _dl_cancel
        with _dl_lock:
            _dl_cancel = True
            for it in _dl_queue:
                if it["status"] == "pending":
                    it["status"] = "failed"
                    it["error"] = "cancelled"
        return {"ok": True}

    @app.post("/api/ingest/register")
    def _register(payload: dict = Body(...)):
        kind = payload.get("kind")
        uri = payload.get("uri")
        if kind not in ("youtube_url", "video_file"):
            raise HTTPException(400, "kind must be youtube_url or video_file")
        if not uri:
            raise HTTPException(400, "uri required")
        title = payload.get("title")
        if kind == "youtube_url" and not title:
            title = get_video_title(uri)
        sid = ingester.register_source(
            kind=kind,
            uri=uri,
            game_version=payload.get("game_version", "fh4"),
            biome_override=payload.get("biome_override"),
            hud_mask=payload.get("hud_mask", []),
            title=title,
        )
        return {"ok": True, "source_id": sid}

    @app.post("/api/ingest/update_mask")
    def _update_mask(payload: dict = Body(...)):
        sid = int(payload["source_id"])
        mask = payload.get("hud_mask", [])
        with write_conn(ingester.db_path) as c:
            c.execute(
                "UPDATE sources SET hud_mask_json=? WHERE id=?",
                (json.dumps(mask), sid),
            )
        return {"ok": True}

    @app.post("/api/ingest/start")
    def _start(payload: dict = Body(...)):
        # If the caller included a hud_mask, persist it before starting
        if "hud_mask" in payload:
            sid = int(payload["source_id"])
            with write_conn(ingester.db_path) as c:
                c.execute(
                    "UPDATE sources SET hud_mask_json=? WHERE id=?",
                    (json.dumps(payload["hud_mask"]), sid),
                )
        ok = ingester.submit(
            source_id=int(payload["source_id"]),
            sample_every_sec=float(payload.get("sample_every_sec", 1.0)),
        )
        if not ok:
            raise HTTPException(409, "another ingest job is running")
        return {"ok": True}

    @app.post("/api/ingest/cancel")
    def _cancel():
        ingester.cancel()
        return {"ok": True}

    @app.get("/api/ingest/status")
    def _status():
        return {"job": ingester.status()}

    @app.post("/api/ingest/probe_video")
    def _probe(payload: dict = Body(...)):
        """Open a video and return a thumbnail + dims for the HUD mask editor.
        Seeks to ~10% into the video to avoid black intro frames."""
        import base64
        path = (payload.get("uri") or "").strip()
        if not path:
            raise HTTPException(400, "uri is empty")
        resolved = Path(path.replace("\\", "/"))
        if not resolved.exists():
            resolved = Path(path)
        if not resolved.exists():
            raise HTTPException(404, f"File not found: {path!r}")
        cap = cv2.VideoCapture(str(resolved).replace("\\", "/"))
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            raise HTTPException(400, f"OpenCV cannot open: {resolved}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # Seek to 10% to skip black intros
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, int(total_frames * 0.10)))
        ret, frame = cap.read()
        # If still black, try other positions
        if not ret or (frame is not None and frame.mean() < 10):
            for pct in (0.20, 0.30, 0.50, 0.05):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, int(total_frames * pct)))
                ret, frame = cap.read()
                if ret and frame.mean() >= 10:
                    break
        cap.release()
        if not ret:
            raise HTTPException(400, "Could not read any frame")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise HTTPException(500, "Failed to encode thumbnail")
        return {
            "width": frame.shape[1],
            "height": frame.shape[0],
            "thumbnail_b64": base64.b64encode(buf.tobytes()).decode(),
        }
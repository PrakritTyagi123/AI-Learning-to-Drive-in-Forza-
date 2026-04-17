# ForzaTek AI

**Custom perception training platform for Forza Horizon 4 / 5 / 6.**

A FastAPI + HTML web app that handles the full machine-learning loop for building a Forza driving perception model: data capture from live gameplay or YouTube videos, an annotation tool for segmentation and object detection, a training pipeline, and an active-learning workflow so the model does most of the labeling work for you after the first few hundred frames.

---

## What this solves

Pretrained models (YOLOv8 on COCO, DeepLabV3 on VOC, etc.) do a mediocre job on Forza. They were trained on real-world driving photos, so they miss game-specific cases: cartoony-looking vehicles, stylized road textures, wet cobblestone that doesn't look like a road to them, etc.

ForzaTek AI trains a **custom multi-task model** on your own labeled Forza frames:
- **Segmentation head** — 4 classes: offroad, road, curb, wall
- **Detection head** — 2 classes: vehicle, sign
- **Shared backbone** — EfficientNet-B1, ImageNet-pretrained, ~12M params total, ~5 ms on an RTX 4070+

The key insight is that labeling 3000 frames by hand is awful. Instead, you **label ~500 frames manually**, train a first model, and then the model labels the rest. You review its output and fix the mistakes. Each round is faster than the one before because the model gets better. This is called **active learning** and it's how labeling actually works in industry.

---

## Quick start

```bash
# Clone, cd into the folder
pip install -r requirements.txt
python -m backend.main        # or: ./scripts/run.sh
# Open http://localhost:8000
```

On Windows, double-click `scripts\run.bat` instead.

First boot is slow because it downloads pretrained weights for the pre-labeler (about 150 MB for YOLOv8n + DeepLabV3). Subsequent starts are instant.

---

## The active learning workflow

This is the part that matters. Read this carefully.

### Round 1 — Bootstrap (manual, ~500 frames)
1. Go to `/record` (for live Forza) or `/ingest` (for YouTube / local videos) and collect 1,500 – 3,000 raw frames.
2. Go to `/label`. Click **PRE-LABEL** to seed the frame with proposals from pretrained models (they're sloppy but save 30 seconds per frame).
3. Correct the segmentation and boxes. Press **Space** to save and move to the next frame.
4. Do this ~500 times. Expect 1–2 minutes per frame at the start, dropping to ~30 seconds as you get into a rhythm.
5. Train the first model:
   ```bash
   python -m scripts.run_pipeline --round 1 --epochs 30
   ```
   This trains for ~20 minutes on a 4080, runs the trained model on 500 random unlabeled frames, and queues the 50 most uncertain ones for you to review.

### Round 2 — Review (~50 frames, fast)
1. Go back to `/label`. It will automatically serve the 50 uncertain frames.
2. For each one, the seg mask and boxes are already drawn by the model.
   - If it looks correct, press **A** (accept).
   - If it's mostly right, make small corrections and press **Space** (saves as edited).
   - If it's garbage, press **R** to clear and label from scratch.
3. Run round 2:
   ```bash
   python -m scripts.run_pipeline --round 2 --epochs 20 --resume models/round_1.pt
   ```

### Round 3 — Scale up (~500 frames, very fast)
Same as round 2, but the model is now much better. Most frames will be correct; you'll mostly be hitting **A** and moving on. Budget 2–3 seconds per frame.

### Round 4+ — Audit mode
At this point the model is strong enough to label everything on its own. Spot-check samples; only correct what's obviously wrong. Run:
```bash
python -m training.predict --use-active --limit 5000 --queue-top-k 0
```
This labels 5,000 frames without queuing any for review — they go straight into `proposals` with provenance `auto_trusted` if you decide to trust them.

### Expected effort

| Round | Frames you touch | Time per frame | Total effort |
|-------|------------------|----------------|--------------|
| 1 | 500 (from scratch) | 60 – 90 s | 8 – 12 hours |
| 2 | 50 (review) | 15 – 30 s | 15 – 30 min |
| 3 | 500 (fast review) | 3 – 10 s | 30 – 80 min |
| 4+ | spot-check only | – | as much as you want |

Total realistic ground truth: about 10 hours to get a solid FH4 model. Fine-tuning for FH5 / FH6 later adds ~1–2 hours each (just 500 new frames per version).

---

## Project layout

```
forzatek/
├── backend/
│   ├── main.py               FastAPI entry point
│   ├── database.py           SQLite schema and helpers
│   ├── recorder.py           Live Forza frame capture (pHash dedup, bucket tracking)
│   ├── video_ingester.py     YouTube + local video → frames (HUD masking, menu filtering)
│   ├── labeling_backend.py   Labeling tool HTTP routes
│   └── prelabeler.py         Pretrained model wrappers (DeepLabV3 + YOLOv8)
├── training/
│   ├── model.py              Multi-task architecture (backbone + seg head + det head)
│   ├── train.py              Training loop with mixed precision, stratified val split
│   └── predict.py            Inference + active-learning queue updater
├── frontend/
│   ├── dashboard.html        Unified dashboard (stats, nav)
│   ├── record-panel.html     Live recording UI
│   ├── ingest-panel.html     Video ingest UI with HUD mask editor
│   └── label-tool.html       Annotation tool (brush + boxes + review mode)
├── scripts/
│   ├── run_pipeline.py       One-shot train+predict orchestrator
│   ├── run.sh                Linux/Mac launcher
│   └── run.bat               Windows launcher
├── tests/
│   ├── test_database.py
│   ├── test_recorder.py
│   └── test_prelabeler.py
├── data/                     Created on first run (SQLite DB + downloaded videos)
├── models/                   Training checkpoints land here
├── requirements.txt
└── README.md
```

---

## Hardware notes

- **Training**: RTX 4070 or better. 8+ GB VRAM is comfortable for batch size 8 at 512×288. 4080/4090 can push batch 16 for faster epochs.
- **Inference (for your driving code)**: A 5070 Ti or similar will run this at 150 – 200 FPS at batch 1, which is overkill for Forza but leaves headroom for your main AI model.
- **Storage**: The SQLite database grows at roughly 100 MB per 1,000 frames (frames are stored as JPEGs inside the DB). 3,000 labeled frames ≈ 300 MB. The `data/videos/` folder can grow fast if you download long YouTube videos; delete them after ingestion.

---

## Common gotchas

**"Not enough labeled frames to train"** — You need at least ~50 frames with *both* segmentation and detection labels before training will run. If you labeled only seg on some frames, they'll be excluded.

**Model predictions look wrong even after round 2** — Check that your labels are consistent. Segmentation especially: if you used "curb" on some frames and "road" on similar-looking pixels on other frames, the model gets confused. Look at the `provenance` column in the `labels` table — if you have lots of `proposed_accepted` labels where you rushed, those might be wrong.

**yt-dlp fails to download** — Update it: `pip install -U yt-dlp`. YouTube breaks yt-dlp every few months.

**HUD mask doesn't seem to apply to YouTube videos** — YouTube downloads happen inside the ingest job. To edit the HUD mask for them, register the source first, then download manually and re-register as a local file. (Or add the HUD mask via the `/api/ingest/update_mask` route before starting.)

**Training crashes with OOM** — Lower `--batch-size` to 4 or 2.

---

## Extending to other games

The database schema has a `game_version` column on every frame. When Horizon 6 ships:
1. Record/ingest ~500 FH6 frames.
2. Label them in the same tool.
3. Fine-tune from the FH5 checkpoint:
   ```bash
   python -m scripts.run_pipeline --round 1 --resume models/round_N.pt
   ```
   Training will include all your FH4 + FH5 + FH6 labels; the new version benefits from everything you labeled before.

---

## Next steps after the perception model is done

Once you have a model you trust, you wire it into the actual driving pipeline:
1. `predict.py` shows how to call the model on a single frame.
2. Feed the segmentation mask to your path planner (drivable pixels = road class).
3. Feed detection boxes to your collision avoidance layer.
4. The trained `.pt` checkpoint can be exported to ONNX for 2× faster inference with TensorRT if you want.

Good luck. Label a few hundred frames, train a first model, and you'll see why this loop works.

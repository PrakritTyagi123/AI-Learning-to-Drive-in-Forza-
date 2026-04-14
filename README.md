# ForzaTek AI Systems v2.0 — Phase 1 + Phase 2

AI-powered telemetry dashboard with live road detection and obstacle overlays for Forza Horizon 4/5/6 and Forza Motorsport.

## Quick Start

### Windows (RTX 4080 recommended)
```
Double-click launch.bat
```
First run downloads PyTorch (~2GB) + YOLOv8 weights (~50MB). Subsequent launches are instant.

### Linux/Mac
```bash
chmod +x launch.sh && ./launch.sh
```

Dashboard: **http://localhost:8080**

## Forza Setup

1. Forza → **Settings → HUD and Gameplay**
2. **DATA OUT** → ON
3. **DATA OUT IP** → `127.0.0.1`
4. **DATA OUT PORT** → `5300`

## What's Included

### Phase 1 — Live Telemetry Dashboard
- Real-time speed, RPM, gear, G-forces, tire data
- Lap timing with delta
- Live game screen capture in viewport
- Track map builder from GPS
- Full engine, suspension, tire traction panels

### Phase 2 — AI Vision Overlays (NEW)
- **YOLOv8-medium** detects cars, trucks, buses, people on screen
- **DeepLabV3-ResNet50** segments the road surface
- **Green dashed lines** = road boundaries extracted from segmentation mask
- **Colored bounding boxes** = obstacles (red=close, amber=medium, blue=far)
- **Purple crosshair** = road center point
- **Toggle switches** below viewport to enable/disable each overlay layer
- **AI inference stats** in the AI Core Brain panel (YOLO ms + SEG ms)

All models are pretrained — zero custom training required. On RTX 4080: ~11ms total inference per frame.

## Architecture

```
forzatek/
├── launch.bat / launch.sh
├── requirements.txt
├── models/                    ← Auto-downloaded on first run
├── backend/
│   ├── server.py              ← Main server (HTTP + WS + UDP + AI)
│   ├── config.py              ← All settings
│   ├── udp_telemetry.py       ← Forza packet parser
│   ├── screen_capture.py      ← mss + OpenCV grabber
│   └── ai_inference.py        ← YOLOv8 + DeepLabV3 inference (NEW)
└── frontend/
    ├── index.html             ← Dashboard with overlay canvas (UPDATED)
    ├── css/dashboard.css      ← Styles with overlay toggles (UPDATED)
    └── js/app.js              ← WS client + overlay renderer (UPDATED)
```

## Testing Without Forza

The dashboard works without Forza running:
1. Launch the server — dashboard loads with empty panels
2. Screen capture shows whatever is on your monitor
3. AI models still run on the captured screen — you'll see road detection on any driving game, YouTube videos of driving, or even Google Street View

## GPU Requirements

| GPU | AI Performance |
|-----|---------------|
| RTX 4080 (recommended) | ~11ms/frame, YOLOv8-medium + DeepLabV3-ResNet50 |
| RTX 3060 | ~20ms/frame, consider YOLOv8-small |
| No GPU | ~200ms/frame on CPU (set AI_DEVICE="cpu" in config.py) |

## Configuration

Edit `backend/config.py`:
- `AI_ENABLED = False` to disable AI inference entirely
- `AI_DEVICE = "cpu"` for CPU-only inference
- `AI_CONFIDENCE_THRESHOLD = 0.4` to adjust detection sensitivity
- `AI_BOUNDARY_SMOOTHING = 5` for smoother/laggier road boundaries

## Ports

| Service | Port |
|---------|------|
| Dashboard | 8080 |
| WebSocket | 8765 |
| Forza UDP | 5300 |

# ForzaTek AI Systems v1.0.4

AI-powered telemetry dashboard for Forza Horizon 4, 5, 6 and Forza Motorsport.

## Quick Start

### Windows
```
Double-click launch.bat
```

### Linux/Mac/WSL
```bash
chmod +x launch.sh
./launch.sh
```

The dashboard opens at **http://localhost:8080**

## Forza Setup (Data Out)

You must enable UDP telemetry in Forza:

1. Open Forza Horizon 5 (or 4/6)
2. Go to **Settings → HUD and Gameplay**
3. Scroll to the bottom
4. Set **DATA OUT** to **ON**
5. Set **DATA OUT IP ADDRESS** to `127.0.0.1`
6. Set **DATA OUT IP PORT** to `5300`
7. Save and start driving!

> For Forza Motorsport, the setting is under **Settings → HUD → Data Out**

## Architecture

```
forzatek/
├── launch.bat / launch.sh    ← One-click launcher
├── requirements.txt
├── backend/
│   ├── server.py             ← Main server (HTTP + WebSocket + UDP)
│   ├── config.py             ← All settings (ports, capture, etc.)
│   ├── udp_telemetry.py      ← Forza packet parser (FH4/5/6 + FM)
│   └── screen_capture.py     ← OpenCV screen grabber
└── frontend/
    ├── index.html             ← Dashboard layout
    ├── css/dashboard.css      ← Dark theme styles
    └── js/app.js              ← WebSocket client + rendering
```

## How It Works

1. **Forza** sends UDP telemetry packets at 60Hz to port 5300
2. **Python backend** receives and parses the 324-byte packets
3. **Screen capture** grabs your monitor at ~30fps using mss + OpenCV
4. **WebSocket server** broadcasts telemetry + frames to the browser
5. **Dashboard** renders everything in real-time with canvas + DOM

## Ports

| Service    | Port | Protocol  |
|------------|------|-----------|
| Dashboard  | 8080 | HTTP      |
| WebSocket  | 8765 | WS        |
| Forza Data | 5300 | UDP       |

## Configuration

Edit `backend/config.py` to change:
- Ports (UDP, WebSocket, HTTP)
- Screen capture settings (FPS, quality, monitor, resolution)
- Game detection

## Telemetry Data Available

All data comes from the Forza UDP "Dash" packet:
- Speed, RPM, Gear, Power, Torque
- Throttle, Brake, Clutch, Handbrake, Steering
- G-Forces (lateral, longitudinal, total)
- Pitch, Yaw, Roll
- Suspension travel (4 wheels)
- Tire slip ratios and angles (4 wheels)
- Tire temperatures (4 wheels)  
- Wheel rotation speeds (4 wheels)
- Car class, Performance Index, Drivetrain, Cylinders
- Lap times, Best lap, Delta, Position
- GPS position (X, Y, Z) for track mapping
- Normalized driving line and AI brake difference

## Phase 2 (Coming)

AI driving model — neural network reads telemetry + screen capture
to predict steering/throttle/brake inputs.

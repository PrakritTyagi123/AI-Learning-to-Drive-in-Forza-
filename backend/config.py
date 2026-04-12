"""
ForzaTek AI — Configuration
All tunable settings in one place.
"""

# ─── Forza UDP Settings ───
UDP_IP = "0.0.0.0"          # Listen on all interfaces
UDP_PORT = 5300              # Forza Data Out port (set this in Forza settings too)

# ─── WebSocket Server ───
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# ─── HTTP Server (frontend) ───
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080

# ─── Screen Capture ───
CAPTURE_ENABLED = True
CAPTURE_FPS = 30             # Target FPS for screen capture
CAPTURE_QUALITY = 50         # JPEG quality (1-100, lower = smaller packets)
CAPTURE_MONITOR = 1          # Monitor index (1 = primary)
CAPTURE_RESIZE = (960, 540)  # Resize captured frame (None = no resize)

# ─── Telemetry ───
TELEMETRY_HZ = 60            # Forza sends at 60Hz, we forward at this rate

# ─── Game Detection ───
SUPPORTED_GAMES = {
    "ForzaHorizon4": "FH4 / FORZA HORIZON 4",
    "ForzaHorizon5": "FH5 / FORZA HORIZON 5",
    "ForzaHorizon6": "FH6 / FORZA HORIZON 6",
    "ForzaMotorsport": "FM / FORZA MOTORSPORT",
}

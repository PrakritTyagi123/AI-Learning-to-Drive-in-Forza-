"""
ForzaTek AI — Configuration
All tunable settings in one place.
"""

# ─── Forza UDP Settings ───
UDP_IP = "0.0.0.0"
UDP_PORT = 5300

# ─── WebSocket Server ───
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# ─── HTTP Server (frontend) ───
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080

# ─── Screen Capture ───
CAPTURE_ENABLED = True
CAPTURE_FPS = 30
CAPTURE_QUALITY = 50
CAPTURE_MONITOR = 1
CAPTURE_RESIZE = (960, 540)

# ─── Telemetry ───
TELEMETRY_HZ = 60

# ─── AI Vision (Phase 2) ───
AI_ENABLED = True                   # Master toggle for AI inference
AI_DEVICE = "cuda"                  # "cuda" for GPU, "cpu" for fallback
AI_ROAD_MODEL = "deeplabv3"         # "deeplabv3" = pretrained, or path to fine-tuned ONNX
AI_OBSTACLE_MODEL = "yolov8m"       # "yolov8m" = pretrained medium, or path to custom .pt
AI_INFERENCE_SIZE = (640, 640)      # YOLO input resolution (higher = more accurate, slower)
AI_SEG_SIZE = (520, 520)            # DeepLabV3 input resolution
AI_CONFIDENCE_THRESHOLD = 0.4       # Min confidence for obstacle detections
AI_ROAD_CLASSES = [0, 1]            # DeepLabV3 VOC class indices: 0=background, we use road pixels
AI_OBSTACLE_CLASSES = [2, 5, 7]     # COCO classes: 2=car, 5=bus, 7=truck
AI_OBSTACLE_EXTRA = [0, 1, 3, 9, 11, 13]  # person, bicycle, motorcycle, traffic light, stop sign, bench
AI_OVERLAY_ENABLED = True           # Send overlay data to frontend
AI_BOUNDARY_SMOOTHING = 5           # Rolling average window for road boundary smoothing

# ─── Game Detection ───
SUPPORTED_GAMES = {
    "ForzaHorizon4": "FH4 / FORZA HORIZON 4",
    "ForzaHorizon5": "FH5 / FORZA HORIZON 5",
    "ForzaHorizon6": "FH6 / FORZA HORIZON 6",
    "ForzaMotorsport": "FM / FORZA MOTORSPORT",
}

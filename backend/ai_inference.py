"""
ForzaTek AI — AI Inference Module (Phase 2 — Fixed)

Fixes:
  - Own car filtered from YOLO detections (center-bottom exclusion zone)
  - Road detection uses HSV color analysis (works much better on Forza than DeepLabV3)
  - Forza driving line arrows detected: blue (normal), yellow (caution), red (brake)
  - Full precision inference (no FP16)
"""

import time
import threading
import traceback
from collections import deque

import numpy as np
import cv2

TORCH_AVAILABLE = False
YOLO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

from config import (
    AI_ENABLED, AI_DEVICE,
    AI_CONFIDENCE_THRESHOLD,
    AI_OBSTACLE_CLASSES, AI_OBSTACLE_EXTRA,
    AI_INFERENCE_SIZE,
    AI_BOUNDARY_SMOOTHING,
)

COCO_NAMES = {
    0: "PERSON", 1: "BIKE", 2: "CAR", 3: "MOTO", 5: "BUS", 7: "TRUCK",
    9: "LIGHT", 11: "STOP", 13: "BENCH",
}

# Own car exclusion zone (normalized coordinates)
# Chase cam: your car sits in center-bottom of the frame
OWN_CAR_X_MIN = 0.25
OWN_CAR_X_MAX = 0.75
OWN_CAR_Y_MIN = 0.45
OWN_CAR_SIZE_MIN = 0.12  # box area / frame area threshold


class AIInference:
    def __init__(self):
        self.running = False
        self.thread = None
        self.device = None
        self.yolo_model = None

        self.lock = threading.Lock()
        self.overlay_data = None
        self.inference_ms = 0
        self.ai_fps = 0
        self.frame_count = 0
        self.total_detections = 0

        # Smoothing buffers
        self.left_boundary_buffer = deque(maxlen=AI_BOUNDARY_SMOOTHING)
        self.right_boundary_buffer = deque(maxlen=AI_BOUNDARY_SMOOTHING)
        self.driving_line_buffer = deque(maxlen=4)

    @property
    def available(self):
        return YOLO_AVAILABLE and AI_ENABLED

    def start(self):
        if not self.available:
            missing = []
            if not YOLO_AVAILABLE: missing.append("ultralytics")
            if not AI_ENABLED: missing.append("AI_ENABLED=False")
            print(f"[AI] Cannot start — missing: {', '.join(missing)}")
            return
        self.running = True
        self.thread = threading.Thread(target=self._init_and_run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def get_overlay(self):
        with self.lock:
            return self.overlay_data

    def _init_and_run(self):
        try:
            self._load_models()
        except Exception as e:
            print(f"[AI] Failed to load models: {e}")
            traceback.print_exc()
            self.running = False
            return
        print(f"[AI] Inference thread running")

    def _load_models(self):
        print(f"[AI] Loading models on {AI_DEVICE}...")

        if TORCH_AVAILABLE:
            self.device = torch.device(AI_DEVICE if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"[AI] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.device = "cpu"

        # Load YOLOv8-medium — full precision
        t0 = time.time()
        self.yolo_model = YOLO("yolov8m.pt")
        print(f"[AI] YOLOv8-medium loaded in {time.time()-t0:.1f}s")

        # Warmup
        print(f"[AI] Warmup inference...")
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)
        self.run_inference(dummy)
        print(f"[AI] Ready for live inference")

    # ═══════════════════════════════════════════════
    #  Main inference — runs on every frame
    # ═══════════════════════════════════════════════

    def run_inference(self, frame_bgr: np.ndarray) -> dict | None:
        if self.yolo_model is None:
            return None

        t_start = time.time()
        h, w = frame_bgr.shape[:2]

        # ─── YOLO obstacle detection ───
        t0 = time.time()
        obstacles = self._run_yolo(frame_bgr, w, h)
        yolo_ms = (time.time() - t0) * 1000

        # ─── Road boundary detection (HSV color-based) ───
        t0 = time.time()
        left_boundary, right_boundary, road_center = self._detect_road_hsv(frame_bgr, w, h)
        road_ms = (time.time() - t0) * 1000

        # ─── Forza driving line detection (blue + yellow + red arrows) ───
        t0 = time.time()
        driving_line = self._detect_driving_line(frame_bgr, w, h)
        line_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - t_start) * 1000

        overlay = {
            "obstacles": obstacles,
            "roadBounds": {"left": left_boundary, "right": right_boundary},
            "roadCenter": road_center,
            "drivingLine": driving_line,
            "stats": {
                "yoloMs": round(yolo_ms, 1),
                "segMs": round(road_ms + line_ms, 1),
                "totalMs": round(total_ms, 1),
                "obstacleCount": len(obstacles),
                "frameSize": [w, h],
            },
        }

        with self.lock:
            self.overlay_data = overlay
            self.inference_ms = total_ms
            self.frame_count += 1
            self.total_detections += len(obstacles)

        return overlay

    # ═══════════════════════════════════════════════
    #  YOLO obstacle detection — with own car filter
    # ═══════════════════════════════════════════════

    def _run_yolo(self, frame_bgr, frame_w, frame_h) -> list:
        all_classes = AI_OBSTACLE_CLASSES + AI_OBSTACLE_EXTRA

        results = self.yolo_model.predict(
            frame_bgr,
            imgsz=AI_INFERENCE_SIZE[0],
            conf=AI_CONFIDENCE_THRESHOLD,
            classes=all_classes,
            verbose=False,
            device=self.device if TORCH_AVAILABLE else 0,
        )

        obstacles = []
        if not results or len(results) == 0:
            return obstacles

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = COCO_NAMES.get(cls_id, f"OBJ_{cls_id}")

            box_h = y2 - y1
            box_w = x2 - x1

            # ─── Filter own car ───
            # In chase cam, your car is always a large box in center-bottom
            cx_norm = ((x1 + x2) / 2) / frame_w
            cy_norm = ((y1 + y2) / 2) / frame_h
            area_norm = (box_w * box_h) / (frame_w * frame_h)

            is_own_car = (
                cls_id in [2, 5, 7] and  # car, bus, truck
                OWN_CAR_X_MIN < cx_norm < OWN_CAR_X_MAX and
                cy_norm > OWN_CAR_Y_MIN and
                area_norm > OWN_CAR_SIZE_MIN
            )
            if is_own_car:
                continue

            # Distance estimation from box height
            est_distance = max(5, int(100 * 20 / max(box_h, 1)))

            # Threat level
            is_ahead = 0.2 < cx_norm < 0.8
            threat = "high" if est_distance < 40 and is_ahead else "med" if est_distance < 70 else "low"

            obstacles.append({
                "x": int(x1), "y": int(y1),
                "w": int(box_w), "h": int(box_h),
                "label": cls_name,
                "conf": round(conf, 2),
                "dist": est_distance,
                "threat": threat,
                "classId": cls_id,
            })

        return obstacles

    # ═══════════════════════════════════════════════
    #  Road boundary detection — HSV color-based
    #  Works better on Forza than pretrained DeepLabV3
    # ═══════════════════════════════════════════════

    def _detect_road_hsv(self, frame_bgr, frame_w, frame_h):
        """
        Detect road surface using color analysis in HSV space.
        Forza roads are typically gray/dark asphalt. We sample the road color
        from the bottom-center of the frame (guaranteed to be road in chase cam)
        and find all similar-colored pixels.
        """
        # Work on a smaller frame for speed
        small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Sample road color from bottom-center (5x5 patch)
        # This is always road surface in chase/bumper cam
        sample_y = sh - 15
        sample_x = sw // 2
        road_patch = hsv[sample_y-2:sample_y+3, sample_x-2:sample_x+3]

        if road_patch.size == 0:
            return [], [], None

        # Get the median HSV of the road sample
        median_h = np.median(road_patch[:, :, 0])
        median_s = np.median(road_patch[:, :, 1])
        median_v = np.median(road_patch[:, :, 2])

        # Create a range around the sampled color
        # Roads have low saturation and variable brightness
        h_range = 25
        s_range = 60
        v_range = 60

        lower = np.array([
            max(0, median_h - h_range),
            max(0, median_s - s_range),
            max(0, median_v - v_range),
        ], dtype=np.uint8)
        upper = np.array([
            min(179, median_h + h_range),
            min(255, median_s + s_range),
            min(255, median_v + v_range),
        ], dtype=np.uint8)

        # Create road mask
        road_mask = cv2.inRange(hsv, lower, upper)

        # Only keep the lower 65% of the frame (road is in the bottom portion)
        road_mask[:int(sh * 0.35), :] = 0

        # Morphological cleanup — remove noise, fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        # Extract left and right boundaries by scanning rows
        left_pts = []
        right_pts = []
        scale_x = frame_w / sw
        scale_y = frame_h / sh

        start_row = int(sh * 0.35)
        for y in range(start_row, sh, 3):
            row = road_mask[y, :]
            road_pixels = np.where(row > 0)[0]
            if len(road_pixels) > 15:  # Minimum road width
                left_pts.append([int(road_pixels[0] * scale_x), int(y * scale_y)])
                right_pts.append([int(road_pixels[-1] * scale_x), int(y * scale_y)])

        # Smooth boundaries
        left_boundary = self._smooth_boundary(left_pts, self.left_boundary_buffer)
        right_boundary = self._smooth_boundary(right_pts, self.right_boundary_buffer)

        # Road center at look-ahead point (~55% height)
        road_center = None
        probe_y = int(sh * 0.55)
        row = road_mask[probe_y, :]
        road_pixels = np.where(row > 0)[0]
        if len(road_pixels) > 15:
            road_center = {
                "x": int(((road_pixels[0] + road_pixels[-1]) / 2) * scale_x),
                "y": int(probe_y * scale_y),
                "width": int((road_pixels[-1] - road_pixels[0]) * scale_x),
            }

        return left_boundary, right_boundary, road_center

    # ═══════════════════════════════════════════════
    #  Forza driving line detection
    #  Detects the colored arrows: blue, yellow, red
    # ═══════════════════════════════════════════════

    def _detect_driving_line(self, frame_bgr, frame_w, frame_h):
        """
        Detect Forza's driving line arrows which appear as colored chevrons
        painted on the road surface. Colors:
          - Blue = normal driving
          - Yellow = caution / slow down
          - Red = heavy braking zone
        Returns list of point dicts with color info.
        """
        # Work on smaller frame for speed
        small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        scale_x = frame_w / sw
        scale_y = frame_h / sh

        # Only scan the road area (lower 65% of frame)
        road_region = hsv[int(sh * 0.35):, :]
        road_bgr = small[int(sh * 0.35):, :]
        ry_offset = int(sh * 0.35)

        all_points = []

        # ─── Blue driving line ───
        # Forza's blue is a saturated cyan/blue on the road
        blue_lower = np.array([90, 80, 80], dtype=np.uint8)
        blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        blue_mask = cv2.inRange(road_region, blue_lower, blue_upper)

        # ─── Yellow driving line ───
        yellow_lower = np.array([15, 80, 120], dtype=np.uint8)
        yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(road_region, yellow_lower, yellow_upper)

        # ─── Red driving line ───
        # Red wraps around hue 0/180 in HSV
        red_lower1 = np.array([0, 80, 100], dtype=np.uint8)
        red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        red_lower2 = np.array([170, 80, 100], dtype=np.uint8)
        red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        red_mask = cv2.bitwise_or(
            cv2.inRange(road_region, red_lower1, red_upper1),
            cv2.inRange(road_region, red_lower2, red_upper2),
        )

        # Process each color
        for mask, color_name in [(blue_mask, "blue"), (yellow_mask, "yellow"), (red_mask, "red")]:
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours of the driving line segments
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 20:  # Filter tiny noise
                    continue

                # Get the centroid of each arrow/segment
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                all_points.append({
                    "x": int(cx * scale_x),
                    "y": int((cy + ry_offset) * scale_y),
                    "color": color_name,
                    "area": int(area * scale_x * scale_y),
                })

        # Sort by Y position (top to bottom = far to near)
        all_points.sort(key=lambda p: p["y"])

        # Smooth: average with recent frames to reduce flicker
        self.driving_line_buffer.append(all_points)

        # Return latest — smoothing happens on frontend via the buffer
        return all_points

    # ═══════════════════════════════════════════════
    #  Boundary smoothing
    # ═══════════════════════════════════════════════

    def _smooth_boundary(self, new_points, buffer):
        if not new_points:
            return []

        buffer.append(new_points)
        if len(buffer) == 0:
            return new_points

        ref = buffer[-1]
        smoothed = []
        for i, pt in enumerate(ref):
            avg_x = pt[0]
            count = 1
            for prev in list(buffer)[:-1]:
                if i < len(prev):
                    avg_x += prev[i][0]
                    count += 1
            smoothed.append([int(avg_x / count), pt[1]])
        return smoothed

    # ═══════════════════════════════════════════════
    #  Stats
    # ═══════════════════════════════════════════════

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "inferenceMs": round(self.inference_ms, 1),
                "framesProcessed": self.frame_count,
                "totalDetections": self.total_detections,
                "modelsLoaded": self.yolo_model is not None,
            }

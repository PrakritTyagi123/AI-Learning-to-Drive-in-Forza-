"""
ForzaTek AI — Main Server (Phase 1 + Phase 2)
Runs four services in one process:
  1. UDP listener (receives Forza telemetry)
  2. WebSocket server (broadcasts to dashboard)
  3. HTTP server (serves frontend files)
  4. Screen capture thread (grabs game screen)
  5. AI inference thread (road segmentation + obstacle detection)  [NEW Phase 2]
"""

import asyncio
import json
import os
import socket
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
from pathlib import Path

import websockets

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    UDP_IP, UDP_PORT,
    WS_HOST, WS_PORT,
    HTTP_HOST, HTTP_PORT,
    CAPTURE_ENABLED,
    AI_ENABLED, AI_OVERLAY_ENABLED,
)
from udp_telemetry import parse_packet
from screen_capture import ScreenCapture
from ai_inference import AIInference

# ─── Globals ───
connected_clients: set = set()
latest_telemetry: dict = {}
lap_history: list = []
last_lap_number: int = 0
best_lap_raw: float = 0.0
capture = ScreenCapture()
ai = AIInference()
start_time = time.time()


# ═══════════════════════════════════════════════
#  HTTP Server
# ═══════════════════════════════════════════════

class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        frontend_dir = str(Path(__file__).parent.parent / "frontend")
        super().__init__(*args, directory=frontend_dir, **kwargs)

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def start_http_server():
    server = HTTPServer((HTTP_HOST, HTTP_PORT), FrontendHandler)
    server.serve_forever()


# ═══════════════════════════════════════════════
#  AI Inference Loop (NEW Phase 2)
# ═══════════════════════════════════════════════

def ai_inference_loop():
    """
    Continuously grabs frames from screen capture and runs AI inference.
    Runs in a dedicated thread. Results are polled by the broadcast loop.
    """
    fps_counter = 0
    fps_timer = time.time()

    while True:
        if not ai.running:
            time.sleep(0.1)
            continue

        frame = capture.get_frame_raw()
        if frame is None:
            time.sleep(0.01)
            continue

        try:
            ai.run_inference(frame)
            fps_counter += 1

            elapsed = time.time() - fps_timer
            if elapsed >= 5.0:
                ai.ai_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

        except Exception as e:
            print(f"[AI] Inference error: {e}")
            time.sleep(0.5)

        # Don't run faster than capture FPS — no point re-processing the same frame
        time.sleep(0.016)  # ~60fps cap


# ═══════════════════════════════════════════════
#  UDP Listener
# ═══════════════════════════════════════════════

async def udp_listener():
    global latest_telemetry, lap_history, last_lap_number, best_lap_raw

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    print(f"[UDP] Listening on {UDP_IP}:{UDP_PORT}")
    print(f"[UDP] Configure Forza: Data Out IP = 127.0.0.1, Port = {UDP_PORT}")

    loop = asyncio.get_event_loop()

    while True:
        try:
            data = await loop.run_in_executor(None, lambda: sock.recv(1024))
            parsed = parse_packet(data)
            if parsed is None:
                continue

            latest_telemetry = parsed

            current_lap_num = parsed.get("lapNumber", 0)
            if current_lap_num > last_lap_number and last_lap_number > 0:
                last_lap_time = parsed.get("lastLapRaw", 0)
                if last_lap_time > 0:
                    delta = 0
                    if best_lap_raw > 0:
                        delta = last_lap_time - best_lap_raw
                    lap_entry = {
                        "num": last_lap_number,
                        "time": parsed.get("lastLap", "--:--.---"),
                        "timeRaw": last_lap_time,
                        "split": f"{'+' if delta >= 0 else ''}{delta:.3f}",
                    }
                    lap_history.append(lap_entry)
                    if len(lap_history) > 20:
                        lap_history = lap_history[-20:]
                    if best_lap_raw <= 0 or last_lap_time < best_lap_raw:
                        best_lap_raw = last_lap_time
            last_lap_number = current_lap_num

        except BlockingIOError:
            await asyncio.sleep(0.001)
        except Exception as e:
            print(f"[UDP] Error: {e}")
            await asyncio.sleep(0.1)


# ═══════════════════════════════════════════════
#  WebSocket Server
# ═══════════════════════════════════════════════

async def ws_handler(websocket):
    connected_clients.add(websocket)
    remote = websocket.remote_address
    print(f"[WS] Client connected: {remote}")

    try:
        async for message in websocket:
            try:
                cmd = json.loads(message)
                # Handle dashboard commands
                if cmd.get("type") == "toggle_ai":
                    print(f"[WS] AI overlay toggle: {cmd.get('enabled')}")
                elif cmd.get("type") == "toggle_overlay_layer":
                    print(f"[WS] Overlay layer toggle: {cmd.get('layer')} = {cmd.get('enabled')}")
            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected: {remote}")


async def broadcast_loop():
    """Broadcasts telemetry + screen capture + AI overlays to all clients."""
    frame_counter = 0
    send_interval = 1.0 / 60
    capture_every = 2

    while True:
        if connected_clients and latest_telemetry:
            payload = {
                "type": "telemetry",
                "data": latest_telemetry,
                "laps": lap_history,
                "timestamp": time.time(),
                "uptime": round(time.time() - start_time, 1),
            }

            # Attach screen capture frame
            frame_counter += 1
            if CAPTURE_ENABLED and frame_counter >= capture_every:
                frame_counter = 0
                frame = capture.get_frame()
                if frame:
                    payload["frame"] = frame
                    payload["captureFps"] = round(capture.fps_actual, 1)

            # ─── NEW Phase 2: Attach AI overlay data ───
            if AI_ENABLED and AI_OVERLAY_ENABLED:
                overlay = ai.get_overlay()
                if overlay:
                    payload["overlays"] = overlay
                    payload["aiStats"] = ai.get_stats()

            message = json.dumps(payload)

            disconnected = set()
            for client in connected_clients.copy():
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            connected_clients.difference_update(disconnected)

        await asyncio.sleep(send_interval)


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def print_banner():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           FORZATEK AI SYSTEMS v2.0.0         ║")
    print("  ║              Phase 1 + Phase 2               ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Dashboard:  http://localhost:{HTTP_PORT}            ║")
    print(f"  ║  WebSocket:  ws://localhost:{WS_PORT}              ║")
    print(f"  ║  UDP Port:   {UDP_PORT}                            ║")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║  Forza Setup:                                ║")
    print(f"  ║    Data Out IP:   127.0.0.1                  ║")
    print(f"  ║    Data Out Port: {UDP_PORT}                        ║")
    print("  ║    Data Out:      ON                         ║")
    print("  ╠══════════════════════════════════════════════╣")
    ai_status = "ENABLED (GPU)" if AI_ENABLED else "DISABLED"
    print(f"  ║  AI Vision:  {ai_status:<33}║")
    print("  ╚══════════════════════════════════════════════╝")
    print()


async def main():
    print_banner()

    # Start HTTP server
    http_thread = Thread(target=start_http_server, daemon=True)
    http_thread.start()
    print(f"[HTTP] Serving frontend at http://localhost:{HTTP_PORT}")

    # Start screen capture
    if CAPTURE_ENABLED and capture.available:
        capture.start()
    elif CAPTURE_ENABLED:
        print("[CAPTURE] Install: pip install mss opencv-python-headless")

    # ─── NEW Phase 2: Start AI inference ───
    if AI_ENABLED:
        ai.start()
        # Wait for models to load before starting inference loop
        time.sleep(1)
        ai_thread = Thread(target=ai_inference_loop, daemon=True)
        ai_thread.start()
        print(f"[AI] Inference loop started")
    else:
        print(f"[AI] AI vision disabled (set AI_ENABLED=True in config.py)")

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[WS] WebSocket server on ws://localhost:{WS_PORT}")

    # Run UDP listener and broadcast loop concurrently
    await asyncio.gather(
        udp_listener(),
        broadcast_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[FORZATEK] Shutting down...")
        capture.stop()
        ai.stop()
        sys.exit(0)
